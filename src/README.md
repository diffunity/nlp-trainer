# Implementation

## Install dependencies

```bash
pip install -r requirements.txxt
```

## Running the code

``` bash
python3 main.py train {path_to_configuration}
```

* `path_to_configuration` should be dot(.)-separated without the `.py` extension. For example, if your configuration is `configs/configs_cola_baseline.py`, your `path_to_configuration` should be `configs.configs_cola_baseline`. If you are using configurations from `configs_lora`, your `path_to_configuration` should be `configs_lora.configs_cola_lora`

* Our code uses wandb to log losses and accuracies during training. Hence, you need to have a wandb api placed in a file named `wandb_api.local` inside the `configs` folder. 

* This command trains the model with the given configuration, saves intermediate checkpoints in the designated path (refer to the provided configuration), and saves the generated output in the `configs` folder.


## Hyperparameter sweep

Use the following command to execute sweep on the hyperparameters. The argument format is the same as above.

``` bash
python3 sweep_train.py train {path_to_configuration}
```
> an example of a `path_to_configuration` is  `sweep_configs.cola_roberta_config_lora`
