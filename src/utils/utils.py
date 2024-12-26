import json
import inspect
import argparse
import importlib

MODEL_REGISTRY = {}
TASK_REGISTRY = {}
TRAINER_REGISTRY = {}

def register_classes(class_obj, registry: dict):
    assert class_obj.__name__ not in registry, "{} has duplicate class object names, this is not permitted!".format(class_obj.__name__)
    registry[class_obj.__name__] = class_obj

    return registry

def register_to(registry):
    def register_to_inner(class_obj):
        nonlocal registry
        register_classes(class_obj, registry)
    return register_to_inner

def make_registry_entry():
    importlib.import_module("models")
    importlib.import_module("tasks")

def get_configs(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def read_config(path):
    class Args():
        built_in = "__"
        def __init__(self, config):
            for k, i in config.__dict__.items():
                if k[:2] == k[-2:] == self.built_in:
                    # clear built-in modules
                    continue
                setattr(self, k, i)
    config = get_configs(path)

    args = dict()
    for name, obj in inspect.getmembers(config):
        if inspect.isclass(obj) and obj.__module__ == config.__name__:
            args[name] = Args(obj)

    return args

def default_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("mode",
                        choices=['train', 'eval', 'infer'],
                        type=str,
    )

    parser.add_argument("--config-path",
                        required=True,
                        type=str,
    )

    parser.add_argument("--wandb-api-path",
                        required=False,
                        type=str,
    )

    return parser.parse_args()
