import math
import os
import random
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .logger import logger
from .misc import CONST


def worker_init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def recurse_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        recurse_freeze(child)


def build_optimizer(params: Iterable, cfg) -> Optimizer:
    if cfg.OPTIMIZER in ["Adam", "adam"]:
        return torch.optim.Adam(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    elif cfg.OPTIMIZER == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.01)),
        )

    elif cfg.OPTIMIZER in ["SGD", "sgd"]:
        return torch.optim.SGD(
            params,
            lr=cfg.LR,
            momentum=float(cfg.get("MOMENTUM", 0.0)),
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIMIZER} not be implemented yet")


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


def seer_cosine_lambda(num_batches: int, cfg):
    accum = int(cfg.get("GRAD_ACCUM", 1))
    w_ep, epochs = int(cfg.get("WARMUP_EPOCHS", 5)), int(cfg.EPOCH)
    if accum > 1:
        warmup, total = w_ep * num_batches // accum + 1, epochs * num_batches // accum + 1
    else:
        warmup, total = w_ep * num_batches, epochs * num_batches

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda, warmup, total


def build_scheduler(optimizer: Optimizer, cfg, num_batches: int = None):
    scheduler = cfg.SCHEDULER
    lr_decay_step = cfg.get("LR_DECAY_STEP", -1)
    tar_scheduler = None

    if str(cfg.get("SCHEDULER_INTERVAL", "epoch")).lower() == "step":
        assert scheduler == "LinearWarmupCosineAnnealingLR", "SCHEDULER_INTERVAL step implements warmup + cosine only"
        assert num_batches, "SCHEDULER_INTERVAL step needs the loader's batches per epoch"
        lam, _, _ = seer_cosine_lambda(num_batches, cfg)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lam)

    if scheduler == "LinearWarmupCosineAnnealingLR":
        tar_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.get("WARMUP_EPOCHS", 5),
            max_epochs=cfg.EPOCH,
            eta_min=float(cfg.get("ETA_MIN", 0.0)),
        )

    elif isinstance(lr_decay_step, list) and scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "MultiStepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )
    elif scheduler in ("ConstantLR", "Constant", "None", None):
        tar_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    else:
        raise NotImplementedError(f"{scheduler} not yet be implemented")

    return tar_scheduler


def clip_gradient(optimizer, max_norm, norm_type):
    """Clips gradients computed during backpropagation to avoid explosion of gradients."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def split_clip_params(optimizer, frozen_ids=(), readout_ids=()):
    main, readout = [], []
    for g in optimizer.param_groups:
        for p in g["params"]:
            if p.grad is None or id(p) in frozen_ids:
                continue
            (readout if id(p) in readout_ids else main).append(p)
    return main, readout


def clip_readout(params, mode, max_norm, norm_type):
    """TRAIN.GRAD_CLIP.READOUT: `per_tensor` = each tensor to max_norm (clip_gradient, LeWM PPP's clip);
    `global` = one norm over the readout."""
    if mode == "per_tensor":
        for p in params:
            clip_grad.clip_grad_norm_(p, max_norm, norm_type)
    elif mode == "global":
        clip_grad.clip_grad_norm_(params, max_norm, norm_type)
    else:
        raise ValueError(f"GRAD_CLIP.READOUT.MODE {mode}")


def setup_seed(seed, conv_repeatable=True):
    """Setup all the random seeds"""
    logger.warning(f"setup random seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


def init_weights(moudle: nn.Module, pretrained=None):
    if pretrained == "" or pretrained is None:
        logger.warning(f"=> Init {type(moudle).__name__} weights in its' backbone and head")
        """
        Add init for other modules
        ...
        """
    elif os.path.isfile(pretrained):
        logger.info(f"=> Loading {type(moudle).__name__} pretrained model from: {pretrained}")
        checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict_old = checkpoint["state_dict"]
            state_dict = OrderedDict()
            for key in state_dict_old.keys():
                if key.startswith("module."):
                    state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                else:
                    state_dict[key] = state_dict_old[key]
        else:
            logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
            raise RuntimeError()
        moudle.load_state_dict(state_dict, strict=False)
        logger.info("=> Loading SUCCEEDED")
    else:
        logger.error(f"=> No {type(moudle).__name__} checkpoints file found in {pretrained}")
        raise FileNotFoundError()
