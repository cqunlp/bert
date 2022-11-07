# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Class Register Module """

import inspect


class ModuleType:
    """Class module type"""

    DATASET = 'dataset'
    DATASET_LOADER = 'dataset_loader'
    DATASET_SIMPLER = 'dataset_simpler'
    LOSS = 'loss'
    OPTIMIZER = 'optimizer'
    WRAPPER = 'wrapper'

    GENERAL = 'general'


class ClassFactory:
    """
    Module class factory
    """
    __registry__ = {}

    @classmethod
    def register(cls, module_type=ModuleType.GENERAL, alias=None):
        """Register class into registry
        Args:
            module_type (ModuleType) :
                module type name, default ModuleType.GENERAL
            alias (str) : class alias

        Returns:
            wrapper
        """

        def wrapper(register_class):
            """Register class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            class_name = alias if alias is not None else register_class.__name__
            if module_type not in cls.__registry__:
                cls.__registry__[module_type] = {class_name: register_class}
            else:
                if class_name in cls.__registry__[module_type]:
                    raise ValueError(
                        "Can't register duplicate class({})".format(class_name))
                cls.__registry__[module_type][register_class.__name__] = register_class
            return register_class

        return wrapper

    @classmethod
    def register_cls(cls, register_class, module_type=ModuleType.GENERAL, alias=None):
        """Register class with type name.

        Args:
            register_class : class need to register
            module_type :  module type name, default ModuleType.GENERAL
            alias : class name
        """
        class_name = alias if alias is not None else register_class.__name__
        if module_type not in cls.__registry__:
            cls.__registry__[module_type] = {class_name: register_class}
        else:
            if class_name in cls.__registry__[module_type]:
                raise ValueError(
                    "Can't register duplicate class ({})".format(class_name))
            cls.__registry__[module_type][register_class.__name__] = register_class
        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """Determine whether class name is in the current type group.

        Args:
            module_type : Module type
            cls_name : class name

        Returns:
            True/False
        """
        if not class_name:
            return module_type in cls.__registry__

        registered = module_type in cls.__registry__ and class_name in cls.__registry__.get(module_type)

        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """Get class

        Args:
            module_type : Module type
            class_name : class name

        Returns:
            register_class
        """
        # verify
        if not cls.is_exist(module_type, class_name):
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(module_type, class_name))

        if not class_name:
            raise ValueError("Can't find class. class type = {}"
                             .format(class_name))

        register_class = cls.__registry__.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=ModuleType.GENERAL,
                              default_args=None):
        """Get instance.
        Args:
            cfg (dict) : Config dict. It should at least contain the key "type".
            module_type : module type
            default_args (dict, optional) : Default initialization arguments.

        Returns:
            obj : The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a Config, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type",'
                f'but got {cfg}\n{default_args}')
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError('default_args must be a dict or None'
                            f'but got {type(default_args)}')

        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                args.setdefault(k, v)

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')

    @classmethod
    def get_instance(cls, module_type=ModuleType.GENERAL,
                     obj_type=None, args=None):
        """Get instance.
        Args:
            module_type : module type
            obj_type : class type
            args (dict) : object arguments

        Returns:
            object : The constructed object
        """
        if obj_type is None:
            raise ValueError("class_name cannot be None.")

        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')
