import os
from datetime import datetime
# roberta
# epochs: 10 w/ early stopping
# lr ∈ {1e−5, 2e−5, 3e−5}
# bsz ∈ {16, 32}
# warmup ratio : 0.06

class task:
    model = "SequenceClassificationLoRA"
    model_name = "FacebookAI/roberta-base"
    task_name = "CoLA"
    lora_r = 8
    lora_alpha = 8

class train:
    learning_rate = 1e-5
    epochs = 2
    weight_decay = 0.01
    report_to = "wandb"
    val_batch = 32
    test_batch = 32
    train_batch = 8
    warmup_ratio = 0.06
    grad_accum = 1
    scheduler = "InverseSqrt"
    max_seq_len = 512    
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.path.basename(os.path.realpath(__file__)).split(".")[0],
    )
    checkpoint_steps = 100000000

class wandb_config:
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val/matthews_correlation"},
        "parameters": {
            "batch_size": {"values": [8, 16, 32]},
            "epochs": {"values": [2,5,10]},
            "lr": {"values": [1e-5, 2e-5, 3e-5]},
            "lora_r":{"values": [6,8,10]},
            "lora_alpha":{"values": [6,8,10]}
        },
    }
    project_name = f"roberta_lora_{task.task_name}_sweep_{datetime.now().strftime('%H_%M_%S_%m%d')}"
    api_key_path = f"{os.path.dirname(os.path.realpath(__file__))}/wandb_api.local"
    api_key = open(api_key_path).readline()

class eval(train):
    # checkpoint = 
    test_batch = 32
    model = "SequenceClassificationModel"
