import sys
import importlib

from custom_classes.custom_trainer import CustomTrainer
from custom_classes.custom_evaluator import CustomEvaluator
from utils import (
    MODEL_REGISTRY,
    TASK_REGISTRY,
    TRAINER_REGISTRY,
    read_config,
    default_parser,
    make_registry_entry,
)

from utils.model_utils import set_seed

def main_train(config_path):
    set_seed(42)
    args = read_config(config_path)
    task_class = TASK_REGISTRY.get(args['task'].task_name)
    model_fn = MODEL_REGISTRY.get(args['task'].model)
    task = task_class(args['task'], args['train'], model_fn)
    trainer = CustomTrainer(task, args.get("wandb_config", None))
    trainer.train(args['train'])

def main_eval(config):
    config_path = f"configs.{config}" if "." not in config else config
    args = read_config(config_path)
    task_class = TASK_REGISTRY.get(args['task'].task_name)
    if args['eval'].from_hf:
        model_fn = MODEL_REGISTRY.get(args['eval'].model)
    else:
        model_fn = MODEL_REGISTRY.get(args['task'].model)

    task = task_class(args['task'], args['eval'], model_fn)
    evaluator = CustomEvaluator(task)
    evaluator.evaluate(args['eval'])

def main_infer(config):
    pass

if __name__=="__main__":
    # assert len(sys.argv) == 3, "define mode (train | eval) and config"
    # print("Executing python3", sys.argv)
    # mode = sys.argv[1]
    # config = sys.argv[2]
    args = default_parser()
    make_registry_entry()
    if args.mode == "train":
        main_train(args.config_path)
    elif args.mode == "eval":
        main_eval(args.config_path)
    elif args.mode == "infer":
        main_infer(args.config_path)
