import contextlib
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
sys.path.append(".")

import argparse

_gpu_pre_parser = argparse.ArgumentParser(add_help=False)
_gpu_pre_parser.add_argument("-g", "--gpu_id", type=str, default=None)
_gpu_pre_arg, _ = _gpu_pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_pre_arg.gpu_id if _gpu_pre_arg.gpu_id is not None else "0"

import random
from datetime import timedelta
from time import time

import lib.models  # noqa: F401  (registers the models)
import numpy as np
import torch
from lib.datasets import create_dataset
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import (build_optimizer, build_scheduler, clip_gradient, clip_readout, setup_seed,
                                 split_clip_params)
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.utils.config import CN


def parse_exp_args():
    _parser = argparse.ArgumentParser(description="Train a model from a config (DDP, one process per GPU).")
    _parser.add_argument("-c", "--cfg", help="config file", type=str, default=None)
    _parser.add_argument("--exp_id", default="default", type=str,
                         help="run name; the run is written to exp/<exp_id>_<timestamp>/. Any value other "
                              "than 'default' or 'debug' requires a clean git tree")
    _parser.add_argument("--resume", help="continue the run in this exp/<run> directory", type=str, default=None)

    _parser.add_argument("-w", "--workers", help="data-loader workers (split over the GPUs)", type=int, default=8)
    _parser.add_argument("--prefetch_factor", help="DataLoader prefetch_factor per worker", type=int, default=2)
    _parser.add_argument("-b", "--batch_size", help="total batch size (overrides TRAIN.BATCH_SIZE)", type=int,
                         default=None)
    _parser.add_argument("--num_epochs", help="number of epochs (overrides TRAIN.EPOCH, which also sets the "
                                              "learning-rate schedule)", type=int, default=None)
    _parser.add_argument("--val_batch_size", help="validation batch size (default: the per-GPU training batch)",
                         type=int, default=None)

    _parser.add_argument("-g", "--gpu_id", type=str, default=None, help="GPU ids, e.g. 0,1,2,3 (sets CUDA_VISIBLE_DEVICES)")
    _parser.add_argument("--snapshot", default=5, type=int,
                         help="save checkpoints/checkpoint_<epoch> every this many epochs (0 = never)")
    _parser.add_argument("--no_eval", action="store_true", help="skip validation")
    _parser.add_argument("--eval_interval", default=5, type=int, help="validate every this many epochs")

    _parser.add_argument("--dist_master_addr", type=str, default="localhost")
    _parser.add_argument("-p", "--dist_master_port", type=str, default="60000")
    _parser.add_argument("--node_rank", default=0, type=int, help="node rank for multi-node training")
    _parser.add_argument("--nodes", default=1, type=int, help="number of nodes")
    _parser.add_argument("--dist_backend", default=None, type=str,
                         help="torch.distributed backend (default: nccl with CUDA, else gloo)")
    arg, custom_arg_string = _parser.parse_known_args()

    if arg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    arg.n_gpus = torch.cuda.device_count()
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return arg, custom_arg_string


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def main_worker(gpu_id: int, cfg: CN, arg: argparse.Namespace, time_f: float):
    rank = arg.n_gpus * arg.node_rank + gpu_id
    torch.distributed.init_process_group(arg.dist_backend,
                                         rank=rank,
                                         world_size=arg.world_size,
                                         timeout=timedelta(hours=8))
    assert rank == torch.distributed.get_rank(), "Something wrong with nodes or gpus"
    torch.cuda.set_device(rank)

    setup_seed(cfg.TRAIN.MANUAL_SEED + rank, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)

    torch.distributed.barrier()

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    sampler_seed = int(cfg.TRAIN.get("SAMPLER_SEED", 0))
    if "SAMPLER_SEED" in cfg.TRAIN:
        logger.warning(f"train sampler seed {sampler_seed} (TRAIN.SAMPLER_SEED)")
    train_sampler = DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True,
                                       seed=sampler_seed)
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=int(arg.workers),
                              prefetch_factor=int(arg.prefetch_factor) if int(arg.workers) > 0 else None,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              worker_init_fn=_init_fn,
                              persistent_workers=True)

    if rank == 0 and not arg.no_eval:
        val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
        val_loader = DataLoader(val_data,
                                batch_size=arg.val_batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False,
                                worker_init_fn=_init_fn)
    else:
        val_loader = None

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS, static_graph=True)

    if hasattr(model.module, "param_groups"):
        optim_params = model.module.param_groups(cfg.TRAIN)
    else:
        optim_params = model.parameters()
    optimizer = build_optimizer(optim_params, cfg=cfg.TRAIN)
    sched_per_step = str(cfg.TRAIN.get("SCHEDULER_INTERVAL", "epoch")).lower() == "step"
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN, num_batches=len(train_loader))

    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
    else:
        epoch = 0

    torch.distributed.barrier()

    best_val_score = float("-inf")

    grad_accum = int(cfg.TRAIN.get("GRAD_ACCUM", 1))
    clip_mode = str(cfg.TRAIN.get("GRAD_CLIP", {}).get("MODE", "per_tensor")).lower() \
        if cfg.TRAIN.GRAD_CLIP_ENABLED else "none"
    assert clip_mode in ("none", "per_tensor", "global", "global_every_backward"), clip_mode
    readout_cfg = cfg.TRAIN.GRAD_CLIP.get("READOUT", None) if clip_mode != "none" else None
    readout_ids = set()
    if readout_cfg is not None:
        assert clip_mode in ("global", "global_every_backward"), "GRAD_CLIP.READOUT needs a global main clip"
        readout_ids = {id(p) for p in model.module.readout_parameters()}
        assert readout_ids, "GRAD_CLIP.READOUT set but the model reports no readout parameters"
        logger.warning(f"GRAD_CLIP.READOUT: {len(readout_ids)} readout tensors clipped on their own "
                       f"({readout_cfg.MODE} {readout_cfg.NORM}), out of the {clip_mode} clip")

    def _frozen_ids():
        return {id(p) for p in model.module.frozen_parameters()} \
            if hasattr(model.module, "frozen_parameters") else set()

    def _clip_global():
        main, _ = split_clip_params(optimizer, _frozen_ids(), readout_ids)
        torch.nn.utils.clip_grad_norm_(main, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

    logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
    for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
        train_sampler.set_epoch(epoch_idx)

        model.train()
        if hasattr(model.module, "on_train_epoch_start"):
            model.module.on_train_epoch_start(epoch_idx)
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            if grad_accum == 1 or bidx % grad_accum == 0:
                optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            preds, loss_dict = model(batch, step_idx, "train")
            loss = loss_dict["loss"]
            if grad_accum > 1:
                loss = loss / grad_accum
            backward_ctx = model.module.backward_context() if hasattr(model.module, "backward_context") \
                else contextlib.nullcontext()
            with backward_ctx:
                loss.backward()
            if clip_mode == "global_every_backward":
                _clip_global()
            if grad_accum > 1 and (bidx % grad_accum != grad_accum - 1):
                continue
            if hasattr(model.module, "on_before_optimizer_step"):
                model.module.on_before_optimizer_step()
            if clip_mode == "per_tensor":
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)
            elif clip_mode == "global":
                _clip_global()
            if readout_ids:
                _, readout = split_clip_params(optimizer, _frozen_ids(), readout_ids)
                clip_readout(readout, str(readout_cfg.MODE), float(readout_cfg.NORM),
                             float(readout_cfg.get("TYPE", cfg.TRAIN.GRAD_CLIP.TYPE)))

            optimizer.step()
            optimizer.zero_grad()
            if sched_per_step:
                scheduler.step()

            trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} "
                                     f"{model.module.format_metric('train')}")

        if not sched_per_step:
            scheduler.step()
        logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        torch.distributed.barrier()
        model.module.on_train_finished(recorder, epoch_idx)

        if not arg.no_eval and val_loader is not None:
            if (epoch_idx % arg.eval_interval == arg.eval_interval - 1 or epoch_idx == 0) and rank == 0:
                logger.info("do validation and save results")
                with torch.no_grad():
                    model.eval()
                    valbar = etqdm(val_loader, rank=rank)
                    for bidx, batch in enumerate(valbar):
                        step_idx = epoch_idx * len(val_loader) + bidx
                        model.module(batch, step_idx, "val")

                model.module.on_val_finished(recorder, epoch_idx)

                if hasattr(model.module, "primary_val_score"):
                    score = model.module.primary_val_score()
                    if score > best_val_score:
                        best_val_score = score
                        recorder.record_best_checkpoint(score)

        torch.distributed.barrier()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    if arg.dist_backend is None:
        arg.dist_backend = "nccl" if torch.cuda.is_available() else "gloo"

    arg.world_size = arg.n_gpus * arg.nodes
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    grad_accum = int(cfg.TRAIN.get("GRAD_ACCUM", 1))
    if grad_accum > 1:
        assert arg.batch_size % grad_accum == 0, (
            f"per-GPU batch {arg.batch_size} is not divisible by TRAIN.GRAD_ACCUM {grad_accum}")
        arg.batch_size //= grad_accum
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    logger.info("====> Use Distributed Data Parallel <====")
    torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)
