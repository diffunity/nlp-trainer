"""
batch_size:16
epochs:10
baseline_alpha:8
baseline_r:10
lr:0.00003
"""


import os
from datetime import datetime
# roberta
# epochs: 10 w/ early stopping
# lr ∈ {1e−5, 2e−5, 3e−5}
# bsz ∈ {16, 32}
# warmup ratio : 0.06

class task:
    model = "SequenceClassificationModel"
    model_name = "FacebookAI/roberta-base"
    task_name = "QQP"

class train:
    learning_rate = 0.00003
    epochs = 10
    weight_decay = 0.01
    report_to = "wandb"
    val_batch = 32
    test_batch = 32
    train_batch = 16
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
    project_name = f"roberta_baseline_{task.task_name}_best_{datetime.now().strftime('%H_%M_%S_%m%d')}"
    experiment_name = project_name
    api_key_path = f"{os.path.dirname(os.path.realpath(__file__))}/wandb_api.local"
    api_key = open(api_key_path).readline()
    resume_from_checkpoint = False

class eval(train):
    # checkpoint = 
    test_batch = 32
    model = "SequenceClassificationModel"
