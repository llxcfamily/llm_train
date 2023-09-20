#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import copy
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Optional, Sequence
import json
import csv
import torch
import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    is_torch_tpu_available,
    set_seed,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import random

from torch.utils.data import Dataset


import torch.utils.data as data
import numpy as np
torch.set_printoptions(threshold=np.inf)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    model_inputs: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids = []
    labels = []
    for model_input in model_inputs:
        messages = model_input.split("<|end_of_turn|>")
        tokenized_text = [1] 
        labeled = [1]
        #print("model_input: {}".format(model_input))
        for message in messages:
            if message.startswith("Human: "):
                new_model_input = message + "<|end_of_turn|>" + "Assistant: "
                tokenize_text = tokenizer(
                    new_model_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation=True,
                )["input_ids"]
                #tokenize_text_1 = tokenizer(
                #    new_model_input,
                #    return_tensors="pt",
                #    padding=False,
                #    max_length=max_seq_length,
                #    truncation=True,
                #)["input_ids"]
                #print("tokenize_text_1: {}".format(tokenize_text_1))
                label = copy.deepcopy(tokenize_text)
                for i in range(len(label)):
                    label[i] = IGNORE_INDEX
                #print("label: {}".format(label))
            elif message.startswith("Assistant: "):
                tmp, new_model_input = message.split("Assistant: ", 1)
                new_model_input = new_model_input + "<|end_of_turn|>"
                tokenize_text = tokenizer(
                    new_model_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation=True,
                )["input_ids"]
                #tokenize_text_1 = tokenizer(
                #    new_model_input,
                #    return_tensors="pt",
                #    padding=False,
                #    max_length=max_seq_length,
                #    truncation=True,
                #)["input_ids"]
                #print("tokenize_text_1: {}".format(tokenize_text_1))
                #print(tokenize_text_1.dtype)
                label = copy.deepcopy(tokenize_text)
            tokenized_text.extend(tokenize_text[1:-1])
            labeled.extend(label[1:-1])
            if len(tokenized_text) >= max_seq_length - 1:
                tokenized_text = tokenized_text[:max_seq_length - 1]
                labeled = labeled[:max_seq_length - 1]
                break
            #print("tokenized_text: {}".format(tokenized_text))
        tokenized_text.append(2)
        labeled.append(2)
        input_ids.append(torch.IntTensor([tokenized_text]).to(torch.int64))
        labels.append(torch.IntTensor([labeled]).to(torch.int64))
        #print("input_ids: {}".format(input_ids))
        #print("labels: {}".format(labels))
        #sys.exit()
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, max_seq_length: int, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        model_inputs = []
        import pandas as pd
        data_file = pd.read_csv(data_path, delimiter='\3', quotechar='\4', quoting =csv.QUOTE_MINIMAL)
        count = 0
        for index, row in data_file.iterrows():
            model_input = row["text"]
            model_inputs.append(model_input)
        random.shuffle(model_inputs)
        #model_inputs = model_inputs[:10000]
        real_max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        self.tokenizer = tokenizer
        self.max_seq_length = real_max_seq_length
        self.model_inputs = model_inputs


        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(model_inputs, tokenizer, self.max_seq_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print(self.input_ids[i])
        #sys.exit()
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key][0] for instance in instances] for key in ("input_ids", "labels"))
        #print("-----------------")
        #print(input_ids)
        #print(input_ids.shape)
        #sys.exit()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


from transformers.trainer_callback import TrainerCallback
PREFIX_CHECKPOINT_DIR="checkpoint"
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        try:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
        except :
            print("delete pytorch_model.bin failed.")
        return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # set_nccl_env()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)
    print(sys.argv)
    print("----------")
    sys.exit()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("-------------zl--------------")
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            pass
            # raise ValueError(
            #     f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            #     "Use --overwrite_output_dir to overcome."
            # )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    #data_files = {}
    #data_files["train"] = data_args.train_file
    #extension = data_args.train_file.split(".")[-1]
    #raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name, add_eos_token=True, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, add_eos_token=True, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype="auto"
        )
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(embedding_size)

        model.resize_token_embeddings(len(tokenizer))
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(embedding_size)
        
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = tokenizer.eos_token
            special_tokens_dict = dict()
            special_tokens_dict["pad_token"] = "<pad>"
            special_tokens_dict["additional_special_tokens"] = ["<|end_of_turn|>"]
        # if tokenizer.eos_token is None:
        #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        # if tokenizer.bos_token is None:
        #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        # if tokenizer.unk_token is None:
        #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=model,
            )
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(embedding_size)
        tokenizer.save_pretrained("new_tokenizer")

        #model.config.use_cache = False

        #old_state_dict = model.state_dict
        #print(old_state_dict)
        #model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        #).__get__(model, type(model))
        #print('--------------------------------------------')
        #print(model.state_dict)

    else:
        raise ValueError(
            "You are instantiating a new model from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here"
        )

    #embedding_size = model.get_input_embeddings().weight.shape[0]
    #print(embedding_size)
    #if len(tokenizer) > embedding_size:
    #    model.resize_token_embeddings(len(tokenizer))

        # use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    if training_args.do_train:
        train_dataset = SupervisedDataset(data_path=data_args.train_file,
                                        max_seq_length=data_args.max_seq_length,
                                        tokenizer=tokenizer)
        #print(train_dataset[0])
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")




        #column_names = raw_datasets["train"].column_names
        #text_column_name = "text" if "text" in column_names else column_names[0]
        #if data_args.max_seq_length > tokenizer.model_max_length:
        #    logger.warning(
        #        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        #        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        #    )
        #max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        #padding = "max_length" if data_args.pad_to_max_length else False
        #padding = "longest"
        #print(padding)
        #print("----------------------------------")
        #def tokenize_function(examples):
            # Remove empty lines
        #    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        #    return tokenizer(examples["text"], padding=padding, truncation=True, max_length=max_seq_length)

        #with training_args.main_process_first(desc="dataset map tokenization"):
        #    tokenized_datasets = raw_datasets.map(
        #        tokenize_function,
        #        batched=True,
        #        num_proc=data_args.preprocessing_num_workers,
        #        remove_columns=[text_column_name],
        #        load_from_cache_file=not data_args.overwrite_cache,
        #        desc="Running tokenizer on dataset line_by_line",
        #    )

        #train_dataset = tokenized_datasets["train"]
        #print(train_dataset[0])
        #train_dataset = TextDataset(path=data_args.train_file,
        #                            max_seq_length=data_args.max_seq_length,
        #                            tokenizer=tokenizer)

    if training_args.do_eval:

        eval_dataset = TextDataset(path=data_args.validation_file,
                                    max_seq_length=data_args.max_seq_length,
                                    tokenizer=tokenizer)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    #mlm=False,
                                                    #pad_to_multiple_of=8)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
 #           print("-----------zl1--------")
            checkpoint = last_checkpoint
#        print("-----------zl2--------")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        #model.save_pretrained("models/llama-lora")


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
