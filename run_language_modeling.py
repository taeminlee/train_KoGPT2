# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import torch
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    AutoTokenizer,
    AutoConfig,
    AutoModelWithLMHead,
    GPT2Config,
    GPT2LMHeadModel,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

from torch.utils.data import Dataset
import gluonnlp as nlp
from utils import get_tokenizer, download

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

tokenizer_path = get_tokenizer()

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}

kogpt2_model_path = download(pytorch_kogpt2['url'],
                      pytorch_kogpt2['fname'],
                      pytorch_kogpt2['chksum'],
                      cachedir='~/kogpt2/')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='kogpt2',
        metadata={
            "help": "The model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch or use SKT-KoGPT2 model."
        },
    )
    model_type: Optional[str] = field(
        default="gpt2",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default='kogpt2', metadata={"help": "Pretrained config name or path if not the same as model_name. "
                                                      "Leave None when you use SKT-KoGPT2 model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if you want to use pre-trained transformer tokenizer."}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_gluonnlp_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "If you want to use gluonnlp BERTSentencetokenizer, set True"}
    )

    max_len: Optional[int] = field(
        default=64, metadata={"help": "If you use gluonnlp tokenizer, set max sequence length"}
    )

    vocab_path: Optional[str] = field(
        default=tokenizer_path,
        metadata={"help": "Vocab path to use for gluonnlp_tokenizer"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether splits your data into chunks, being careful not to overstep line returns as each line is interpreted as a document." 
                  "Set True/False based on what your corpus looks like."},
    )

    text_dataset: bool = field(
        default=False,
        metadata={"help": "Whether splits your data into chunks with no attention whatsoever to the line returns or separators." 
                  "Set True/False based on what your corpus looks like."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

class Get_dataset(Dataset):
    def __init__(self, dataset, sent_idx, tokenizer,
                 max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return torch.tensor(self.sentences[i][0], dtype=torch.long)

    def __len__(self):
        return (len(self.sentences))


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, max_len, evaluate=False):

    file_path = args.eval_data_file if evaluate else args.train_data_file

    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    elif args.text_dataset:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )
    else:
        """
        When use common tab separated text dataset, use nlp.data.TSVDataset.
        If you want to use other type of dataset, refer to other class of nlp.data,
        or set DataTrainingArguments.line_by_line or DataTrainingArguments.text_dataset True.
        """
        dataset = nlp.data.TSVDataset(file_path, field_indices=[1],
                                            num_discard_samples=1)

        return Get_dataset(dataset, 0, tokenizer, max_len, True, False)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("training_args: ", training_args)

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        if model_args.config_name == 'kogpt2':
            config = GPT2Config.from_dict(kogpt2_config)
        else:
            config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        if model_args.model_name_or_path == 'kogpt2':
            config = GPT2Config.from_dict(kogpt2_config)
        else:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name and model_args.use_gluonnlp_tokenizer == False:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path and model_args.use_gluonnlp_tokenizer == False:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    elif model_args.use_gluonnlp_tokenizer == True:
        vocab_file = model_args.vocab_path
        vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                       mask_token=None,
                                                       sep_token=None,
                                                       cls_token=None,
                                                       unknown_token='<unk>',
                                                       padding_token='<pad>',
                                                       bos_token='<s>',
                                                       eos_token='</s>')

        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        if model_args.model_name_or_path == 'kogpt2':
            model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=None,
                                                    config=config,
                                                    state_dict=torch.load(kogpt2_model_path))
        else:
            model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    if model_args.use_gluonnlp_tokenizer == False:
        model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if model_args.use_gluonnlp_tokenizer == True:
        max_len = model_args.max_len
    else:
        max_len = tokenizer.max_len

    if data_args.block_size <= 0:
        data_args.block_size = max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, max_len)



    # Get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer, max_len=max_len) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, max_len=max_len, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        print("model_path: ",model_path)
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master() and model_args.use_gluonnlp_tokenizer == False:
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
