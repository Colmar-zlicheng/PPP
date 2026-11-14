import inspect
import warnings
from functools import partial

from .config import CN
from .misc import is_seq_of


def build_from_cfg(cfg, registry, **kwargs):
    """Build a module from config dict."""
    assert isinstance(cfg, CN) and cfg.TYPE is not None

    if not (isinstance(kwargs, dict) or kwargs is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(kwargs)}')

    if type(kwargs) is dict:
        kwargs = {k.upper(): v for k, v in kwargs.items()}  # capitalize all keys for later use as CN
        _cfg = cfg.clone()
        _cfg.defrost()
        _cfg.set_new_allowed(True)
        kwarg_cfg = CN(kwargs)
        _cfg.merge_from_other_cfg(kwarg_cfg)
        _cfg.freeze()
        cfg = _cfg.clone()

    obj_type = cfg.TYPE

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError("{} is not in the {} registry".format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError("type must be a str or valid type, but got {}".format(type(obj_type)))

    return obj_cls(cfg)


class Registry:
    """A registry to map strings to classes."""

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry."""
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key."""
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record."""
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry."""

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(f'The old API of register_module(module, force=False) '
                      f'used in {cls.__name__}'
                      f'is deprecated and will be removed, please use the new API '
                      f'register_module(name=None, force=False, module=None) instead.')
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module."""
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError('name must be either of None, an instance of str or a sequence'
                            f'  of str, but got {type(name)}')

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


MODEL = Registry("model")
DATASET = Registry("dataset")


def build_dataset(cfg: CN, **kwargs):
    return build_from_cfg(cfg, DATASET, **kwargs)


def build_model(cfg: CN, **kwargs):
    return build_from_cfg(cfg, MODEL, **kwargs)
