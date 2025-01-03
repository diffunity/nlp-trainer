

import torch
from torch.utils.data.dataloader import DataLoader

from evaluate import load
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from utils import register_to, TASK_REGISTRY
from utils.qa_utils import postprocess_qa_predictions

# GENE ADDED
from functools import partial
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, make_eval_dict

class TaskClass:

    def __init__(self, task_args, train_args, model_fn):
        self.train_args = train_args
        self.task_args = task_args
        self.tokenizer = AutoTokenizer.from_pretrained(task_args.model_name)
        if getattr(train_args, "from_hf", None):
            task_args.model_name = train_args.checkpoint
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.init_model(model_fn, task_args)

    def init_model(self):
        raise NotImplementedError

    @staticmethod
    def process_function(examples, tokenizer, input_fields):
        raise NotImplementedError

    def loss_function(self, hypo, targ):
        raise NotImplementedError
    
    def prepare(self):
        raise NotImplementedError

    def prepare_eval(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def extract_answer_from_output(self, outp):
        raise NotImplementedError

    def extract_label_from_input(self, inp):
        raise NotImplementedError

    def compute_metric(self, preds, labels):
        raise NotImplementedError

    def print_model_params(self):
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
        print(f"Total params {total_params} | Trainable params {trainable_params} ({trainable_params/total_params})")

@register_to(TASK_REGISTRY)
class SQuADv2(TaskClass):

    def __init__(self, task_args, train_args, model_fn):
        super().__init__(task_args, train_args, model_fn)
        self.criterion = torch.nn.functional.cross_entropy
        self.metric = make_eval_dict # load("squad") # gives an error: 
        # ValueError: Predictions and/or references don't match the expected format.
        # Expected format: {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None), 'no_answer_probability': Value(dtype='float32', id=None)}, 'references': {'id': Value(dtype='string', id=None), 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}},

    @staticmethod
    def process_function(examples, tokenizer, max_seq_len=384):
        questions = [q.strip() for q in examples["question"]]
        
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_seq_len,
            truncation=True if max_seq_len <= 512 else "only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)
                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1
                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)
                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def init_model(self, model_fn, task_args):
        if getattr(task_args, "lora_r") is None or getattr(self.train_args, "from_hf", None):
            self.model = model_fn(task_args.model_name)
        else:
            self.model = model_fn(
                task_args.model_name,
                lora_r=task_args.lora_r,
                lora_alpha=task_args.lora_alpha,
            )

    def prepare(self):
        squad = load_dataset("rajpurkar/squad_v2")
        # GENE: process_function has 3 params so we need an additional wrapper for max_seq_len
        process_with_params = partial(self.process_function, tokenizer=self.tokenizer, max_seq_len=self.train_args.max_seq_len)
        tokenized_squad = squad.map(
            lambda x: process_with_params(x),
            batched=True,
            remove_columns=squad["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_squad['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_squad['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        # test_dataloader = DataLoader(
        #     tokenized_squad['test'],
        #     shuffle=False,
        #     collate_fn=self.data_collator,
        #     batch_size=self.train_args.test_batch,
        # )
        return (
            train_dataloader,
            validation_dataloader,
            None,
        )

    def loss_function(self, hypo, targ):
        # hypo.shape == (bsz, 2, seq_len)
        # targ.shape == (bsz)
        return hypo.loss

    def extract_answer_from_output(self, outp):
        # Extracts the most probable start and end logits from the output
        logit_ans = torch.stack([
                        outp.start_logits.argmax(dim=1),
                        outp.end_logits.argmax(dim=1)
                    ], dim=1).tolist()
        return logit_ans

    def extract_label_from_input(self, inp):
        # Extracts the actual start and end logits from the input
        label_ans = torch.stack([
                        inp['start_positions'],
                        inp['end_positions']
                    ], dim=1).tolist()
        return label_ans

    def compute_metric(self, preds, labels):
        exact_scores = sum([1 if p[0] == a[0] and p[1] == a[1] else 0 for p, a in zip(preds, labels)])
        
        # FIX f1_scores
        f1_scores = [1 if p[0] == a[0] and p[1] == a[1] else 0 for p, a in zip(preds, labels)]
        average_f1 = sum(f1_scores) / len(f1_scores)

        metric = {
            "exact_scores": exact_scores,
            "f1_scores": average_f1
        }
        
        return metric

    def inference(self, inp):
        outp = self.model(**inp)
        return self.extract_answer_from_output(outp)

    def evaluate(self, inp, label):
        pred = self.inference(inp)
        return pred, label.detach().tolist()

class SequenceClassification(TaskClass):

    def __init__(self, task_args, train_args, model_fn):
        super().__init__(task_args, train_args, model_fn)
        self.criterion = torch.nn.functional.cross_entropy
        self.metric = load("glue", self.task_args.task_name.lower())

    def init_model(self, model_fn, task_args):
        if getattr(task_args, "lora_r", None) is None or getattr(self.train_args, "from_hf", None):
            self.model = model_fn(task_args.model_name, num_labels=2)
        else:
            self.model = model_fn(
                task_args.model_name,
                lora_r=task_args.lora_r,
                lora_alpha=task_args.lora_alpha,
                num_labels=2
            )

    @staticmethod
    def process_function(examples, tokenizer, input_fields, max_seq_len=None):
        max_seq_len = 384 if max_seq_len is None else max_seq_len
        if len(input_fields) == 1:
            inp = tokenizer(
                [i.strip() for i in examples[input_fields[0]]],
                max_length=max_seq_len,
                truncation=True,
            )
        else:
            inp = tokenizer(
                [i.strip() for i in examples[input_fields[0]]],
                [i.strip() for i in examples[input_fields[1]]],
                max_length=max_seq_len,
                truncation=True,   
            )
        inp["label"] = examples["label"]
        return inp

    def loss_function(self, hypo, targ):
        # hypo.shape == (bsz, num_classes)
        # targ.shape == (bsz)
        return hypo.loss

    def extract_answer_from_output(self, outp):
        return outp.logits.argmax(dim=1).detach().tolist()

    def extract_label_from_input(self, inp):
        return inp['labels'].detach().tolist()

    def inference(self, inp):
        outp = self.model(**inp)
        return self.extract_answer_from_output(outp)

    def compute_metric(self, preds, labels):
        return self.metric.compute(
            predictions=preds,
            references=labels,
        )

    def evaluate(self, inp, label):
        pred = self.inference(inp)
        return pred, label.detach().tolist()

@register_to(TASK_REGISTRY)
class MNLI(SequenceClassification):

    def init_model(self, model_fn, task_args):
        if getattr(task_args, "lora_r") is None or getattr(self.train_args, "from_hf", None):
            self.model = model_fn(task_args.model_name, num_labels=3)
        else:
            self.model = model_fn(
                task_args.model_name,
                lora_r=task_args.lora_r,
                lora_alpha=task_args.lora_alpha,
                num_labels=3
            )

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test_matched")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
            x, self.tokenizer, ["premise", "hypothesis"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_matched_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_matched_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ["premise", "hypothesis"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            # remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_matched_dataloader = DataLoader(
            tokenized_ds['validation_matched'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test_matched']['idx']
        test_matched_dataloader = DataLoader(
            tokenized_ds['test_matched'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation_matched", "test_matched"
        # ['premise', 'hypothesis', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1, 2
        return (
            train_dataloader,
            validation_matched_dataloader,
            test_matched_dataloader,
        )

@register_to(TASK_REGISTRY)
class SST2(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ["sentence"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ["sentence"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # stanford sentiment treebank (sst2) tests for sentiment (pos/neg) of given sentence

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class MRPC(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # microsoft research paraphrase corpus (mrpc)mtests for semantic equivalence 

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class CoLA(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # tests whether the given sentence is grammatically correct english

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class QNLI(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question', 'sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question', 'sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['question', 'sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # tests for whether the answer to the question can be found in the question
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class QQP(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question1', 'question2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question1', 'question2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['question1', 'question2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # quora question pairs (qqp) tests for semantic equivalence 
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class RTE(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # recognizing textual entailment (rte) tests textual entailment (collapses neutral & contradiction into not entailment)
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class STSB(SequenceClassification):

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        column_names = ds["train"].column_names
        column_names.remove("label")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'].remove_columns(column_names),
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        self.test_idx = tokenized_ds['test']['idx']
        test_dataloader = DataLoader(
            tokenized_ds['test'].remove_columns(column_names),
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: floating point from 0 to 5
        # pair is human-annotated with a similarity score from 1 to 5
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )
