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


def create_vocab(filepath: str) -> None:
    tf.get_logger().setLevel('ERROR')
    pwd = pathlib.Path.cwd()

    # Pre-processing
    # Ajouter la longueur
    # ajouter la possibilité de jouer avec plusieurs .csv et donc plusieurs datasets # suppress warnings

    dataset = tf.data.experimental.CsvDataset(
    filepath,
    [tf.string,  # Required field, use dtype or empty tensor
    tf.string  # Required field, use dtype or empty tensor
    ])
          
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

    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        train_packet.batch(1000).prefetch(2),
        **bert_vocab_args
    )

    with open("./data/vocabulary.txt", 'w') as f:
        for token in vocab:
            print(token, file=f)