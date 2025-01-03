import os
import sys
import wandb
import importlib

from custom_classes.custom_trainer import CustomTrainer
from utils import (
    MODEL_REGISTRY,
    TASK_REGISTRY,
    TRAINER_REGISTRY,
    read_config,
    set_seed,
)

importlib.import_module("model")
importlib.import_module("task")

def main(config_path):
    set_seed(42)
    args = read_config(config_path)
    task_class = TASK_REGISTRY.get(args['task'].task_name)
    model_fn = MODEL_REGISTRY.get(args['task'].model)

    wandb_config = args['wandb_config']
    wandb.login(key=wandb_config.api_key)
    sweep_id = wandb.sweep(
        sweep=wandb_config.sweep_configuration,
        project=wandb_config.project_name,
        # Track hyperparameters and run metadata
    )
    args['train'].checkpoint_path_parent = args['train'].checkpoint_path

    def sweep_function():

        wandb.init()

        print("Sweep run name", wandb.config._settings.run_name)
        # note that we define values from `wandb.config`
        # instead of defining hard values
        args['train'].learning_rate = wandb.config.lr
        args['train'].train_batch = wandb.config.batch_size
        args['train'].epochs = wandb.config.epochs
        args['task'].lora_r = wandb.config.lora_r
        args['task'].lora_alpha = wandb.config.lora_alpha
        args['train'].checkpoint_path = os.path.join(
            args['train'].checkpoint_path_parent, f"config_{wandb.config._settings.run_name}")
        print("save checkpoint to ", args['train'].checkpoint_path)
        task = task_class(args['task'], args['train'], model_fn)
        trainer = CustomTrainer(task, args.get("wandb_config", None), sweep=True)
        trainer.train(args['train'])

    wandb.agent(sweep_id, function=sweep_function, count=7)

if __name__=="__main__":
    assert len(sys.argv) == 3, f"{sys.argv} define mode (train | eval) and config"
    print("Executing python3", sys.argv)
    mode = sys.argv[1]
    config = sys.argv[2]
    main(config)
