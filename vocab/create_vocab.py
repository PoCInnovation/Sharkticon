import collections
import os
import pathlib
import re
import string
import pandas as pd
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

# Pre-processing
# Ajouter la longueur
# ajouter la possibilité de jouer avec plusieurs .csv et donc plusieurs datasets # suppress warnings

dataset = tf.data.experimental.CsvDataset(
  '../Dataset_IDS.csv',
  [tf.string,  # Required field, use dtype or empty tensor
   tf.string  # Required field, use dtype or empty tensor
  ])

for input_packet, target_packet in dataset.batch(3).take(1):
  print(input_packet)
  for pckt in input_packet.numpy():
    print(pckt.decode('utf-8'))

  print()

  for packet in target_packet.numpy():
    print(packet.decode('utf-8'))

  
for packet in input_packet.numpy():
  print(packet.decode('utf-8'))
  
train_packet = dataset.map(lambda packet, target: target)
train_target = dataset.map(lambda packet, target: packet)

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 30000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

print('starting tokenization ...')

pt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_packet.batch(1000).prefetch(2),
    **bert_vocab_args
)

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

print('last step, writing in a file')

write_vocab_file('vocab.txt', pt_vocab)