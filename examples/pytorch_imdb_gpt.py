#!/usr/bin/env python
# coding: utf-8

# # IMDB movie review text generation
#
# In this script, we'll fine-tune a GPT2-like model to generate more
# movie reviews based on a prompt.
#
# Partly based on this tutorial:
# https://github.com/omidiu/GPT-2-Fine-Tuning/

import torch
import os
import sys
import math

from pprint import pprint
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)
from transformers import pipeline

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')


datapath = os.getenv('DATADIR')
if datapath is None:
    print("Please set DATADIR environment variable!")
    sys.exit(1)
user_datapath = os.path.join(datapath, "users", os.getenv('USER'))
os.makedirs(user_datapath, exist_ok=True)


# ## IMDB data set
#
# Next we'll load the IMDB data set, this time using the Hugging Face
# datasets library: https://huggingface.co/docs/datasets/index.
#
# The dataset contains 100,000 movies reviews from the Internet Movie
# Database, split into 25,000 reviews for training and 25,000 reviews
# for testing and 50,000 without labels (unsupervised).

train_dataset = load_dataset("imdb", split="train+unsupervised")
eval_dataset = load_dataset("imdb", split="test")

# Let's print one sample from the dataset.
print('Sample from dataset')
for b in train_dataset:
    pprint(b)
    break


# We'll use the distilgpt2 model from the Hugging Face library:
# https://huggingface.co/distilgpt2
# Let's start with getting the appropriate tokenizer.

pretrained_model = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
special_tokens = tokenizer.special_tokens_map


# We'll add the special token for indicating the end of the text at
# the end of each review
def apply_transform(x):
    return tokenizer(x['text'] + special_tokens['eos_token'], truncation=True)


train_dataset_tok = train_dataset.map(apply_transform,
                                      remove_columns=['text', 'label'])
eval_dataset_tok = eval_dataset.map(apply_transform,
                                    remove_columns=['text', 'label'])


print('Sample of tokenized data')
for b in train_dataset_tok:
    pprint(b, compact=True)
    print('Length of input_ids:', len(b['input_ids']))
    break

print('Length of dataset (tokenized)', len(train_dataset_tok))


# Next, we'll group the variable-length tokenized texts into
# fixed-length blocks for more efficient processing.
#
# - We'll feed 1000 reviews (batch_size=1000 below) to the function in
# one "batch".
#
# - Each review has a variable length of text, each ending with an
# end-of-sentence token.
#
# - We concatenate these texts and split them up to equal length (128)
# blocks
#
def divide_tokenized_text(tokenized_text_dict, block_size):
    # join the 'input_ids' and 'attention_mask' fields from all texts
    # into a single dictionary which has the items concatenated
    concatenated_examples = {k: sum(tokenized_text_dict[k], []) for k in tokenized_text_dict.keys()}

    # total length of concatenated lists (total number of words)
    total_length = len(concatenated_examples[list(tokenized_text_dict.keys())[0]])

    # we leave out the last items so that we get blocks of equal size block_size
    total_length = (total_length // block_size) * block_size

    # dictionary results will have same fields, 'inputs_ids' and
    # 'attention_mask', but these will now contain lists of blocks,
    # each with block_size items
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result['labels'] = result['input_ids'].copy()
    return result


max_block_length = 128

train_dataset_batched = train_dataset_tok.map(
    lambda tokenized_text_dict: divide_tokenized_text(tokenized_text_dict, max_block_length),
    batched=True,
    batch_size=1000,
    num_proc=4,
)

eval_dataset_batched = eval_dataset_tok.map(
    lambda tokenized_text_dict: divide_tokenized_text(tokenized_text_dict, max_block_length),
    batched=True,
    batch_size=1000,
    num_proc=4,
)


print('Sample of grouped text')
for b in train_dataset_batched:
    pprint(b, compact=True)
    print('Length of input_ids:', len(b['input_ids']))
    break
print('Length of dataset (grouped)', len(train_dataset_batched))

# Randomize order of training set
train_dataset_shuffled = train_dataset_batched.shuffle(seed=42)

# Load the actual base model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Here we use the Hugging Face trainer instead of our own training
# function

# You can read about the many, many different parameters to the
# Hugging Face trainer here:
# https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments

output_dir = os.path.join(user_datapath, "gpt-imdb-model")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,
    weight_decay=0.01,
    # per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    max_steps=5000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset_shuffled,
    eval_dataset=eval_dataset_batched,
)

trainer.train()

print()
print("Training done, you can find all the model checkpoints in", output_dir)

# Calculate perplexity
eval_results = trainer.evaluate()
print(f'Perplexity: {math.exp(eval_results["eval_loss"]):.2f}')

# Let's print a few sample generated reviews
prompt = "This movie was awful because"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
outputs = model.generate(input_ids, do_sample=True, max_length=80, num_return_sequences=4)
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print('Sample generated review:')
for txt in decoded_outputs:
    print('-', txt)
