!pip install tokenizers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
import json
import time

# Pre-processing
# Ajouter la longueur
dataset = [["192.168.1.1", "3000", "192.168.2.3:30", "3500", "TCP"]]

def concat_dataset(dataset):
  tokens = []
  for i in dataset:
    print(i)
    tokens.append(' '.join(i))
  return tokens

print(concat_dataset(dataset))

class PacketTokenizer():
  def __init__(self, path_to_vocab):
    self._vocab = path_to_vocab
    self._tokenizer = Tokenizer.from_file(self._vocab)


  def train(self, files):
    self._tokenizer.train(
                    files,
                    vocab_size=100,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=['[START]', '[END]', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                    limit_alphabet=1000,
                    wordpieces_prefix="##"
                  )

  def tokenize(self, msg):
    token = self._tokenizer.encode("[START] "  + msg + " [END]")
    return token

  def detokenize(self, tokens):
    return self._tokenize.decode(tokens.ids)

  def get_vocab_path(self):
    return self._tokenizer

  def get_vocab_size(self):
    return self._tokenizer.get_vocab_size()

  def vocab(self):
    try:
      with open(f"{self._vocab}", "r") as vocab:
        data = json.load(vocab)
      print(data)
    except Exception:
      print("No Vocab file accessible")
    return data


  def lookup(self, packet):
    return tf.ragged.constant(self.tokenize(packet))

  def tokenizer(self):
    pass


data = [["192.168.1.1", "3000", "192.168.2.3", "3500", "17"]]

packetTokenizer = PacketTokenizer("./my-tokenizer.json")
  packetTokenizer.vocab()
  print(packetTokenizer.tokenize(concat_dataset(data)))
