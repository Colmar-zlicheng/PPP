from argparse import Namespace

from .logger import logger
from yacs.config import CfgNode as _CN
from copy import deepcopy


class CN(_CN):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.recursive_cfg_update()

    def recursive_cfg_update(self):
        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        return self

    def dump(self, *args, **kwargs):
        def change_back(cfg: CN) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return _CN(cfg).dump(*args, **kwargs)


_C = CN(new_allowed=True)

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.MANUAL_SEED = 1
_C.TRAIN.CONV_REPEATABLE = True
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCH = 100
_C.TRAIN.OPTIMIZER = "Adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.SCHEDULER = "StepLR"
_C.TRAIN.LR_DECAY_GAMMA = 0.1
_C.TRAIN.LOG_INTERVAL = 50
_C.TRAIN.FIND_UNUSED_PARAMETERS = False
_C.TRAIN.GRAD_CLIP_ENABLED = True
_C.TRAIN.GRAD_CLIP = CN(new_allowed=True)
_C.TRAIN.GRAD_CLIP.TYPE = 2
_C.TRAIN.GRAD_CLIP.NORM = 0.001


def default_config() -> CN:
    """Get a yacs CfgNode object with the default config values."""
    return _C.clone()


def get_config(config_file: str, arg: Namespace = None, merge: bool = True, freeze=True) -> CN:
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    if arg is not None:
        if arg.batch_size is not None:
            if arg.resume is None:
                logger.warning(f"cfg's batch_size {cfg.TRAIN.BATCH_SIZE} reset to arg.batch_size: {arg.batch_size}")
            cfg.TRAIN.BATCH_SIZE = arg.batch_size
        else:
            arg.batch_size = cfg.TRAIN.BATCH_SIZE

        if arg.num_epochs is not None:
            if arg.resume is None:
                logger.warning(f"cfg's TRAIN.EPOCH {cfg.TRAIN.EPOCH} reset to arg.num_epochs: {arg.num_epochs}")
            cfg.TRAIN.EPOCH = arg.num_epochs
        else:
            arg.num_epochs = cfg.TRAIN.EPOCH

    if freeze:
        cfg.freeze()
    return cfg

