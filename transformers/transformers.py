import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

BUFFER_SIZE = 20000
BATCH_SIZE = 64

logging.getLogger('tensorflow').setLevel(logging.ERROR)

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

for pt_examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    print(en.decode('utf-8'))

model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load(model_name)
[item for item in dir(tokenizers.en) if not item.startswith('_')]
for en in en_examples.numpy():
  print(en.decode('utf-8'))

encoded = tokenizers.en.tokenize(en_examples)

for row in encoded.to_list():
  print(row)

round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
  print(line.decode('utf-8'))

tokens = tokenizers.en.lookup(encoded)
tokens

def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
temp

# n, d = 2048, 512
# pos_encoding = positional_encoding(n, d)
# print(pos_encoding.shape)
# pos_encoding = pos_encoding[0]

# # Juggle the dimensions for the plot
# pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
# pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
# pos_encoding = tf.reshape(pos_encoding, (d, n))

# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.ylabel('Depth')
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()

exit()

input_embedding = [[
    "Salut", "comment", "ça", "va", "?"
]]

output_embedding = [[
   "<START>", "Hi", "how", "are", "you", "?"
]]

def get_vocabulary(sequences):

    token_to_info = {}

    for sequence in sequences:
        for word in sequence:
            if word not in token_to_info:
                token_to_info[word] = len(token_to_info)
    return token_to_info

input_voc = get_vocabulary(input_embedding)
output_voc = get_vocabulary(output_embedding)

input_voc["<START>"] = len(input_voc)
input_voc["<END>"] = len(input_voc)
input_voc["<PAD>"] = len(input_voc)

output_voc["<END>"] = len(output_voc)
output_voc["<PAD>"] = len(output_voc)

print(input_voc)
print(output_voc)

def sequence_to_int(sequences, voc):
    for sequence in sequences:
        for s, word in enumerate(sequence):
            sequence[s] = voc[word]
    return np.array(sequences)

input_seq = sequence_to_int(input_embedding, input_voc)
output_seq = sequence_to_int(output_embedding, output_voc)
print(input_seq)
print(output_seq)

class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, nb_token, **kwargs):
        self.nb_token = nb_token
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.word_embedding = tf.keras.layers.Embedding(self.nb_token, 256)

    def call(self, x):
        embed = self.word_embedding(x)
        return embed


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(256)
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        super().build(input_shape)

    def call(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(256.)
        softmax_QK = tf.nn.softmax(QK, axis=-1)
        attention = tf.matmul(softmax_QK, V)

        return attention

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, dim=256, nb_head=8, **kwargs):
        self.dim = dim
        self.head_dim = dim // nb_head
        self.nb_head = nb_head
        super(**kwargs).__init__()

    def build(self, input_shape):

        self.query_layer = tf.keras.layers.Dense(256)
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        self.out_proj = tf.keras.layers.Dense(256)

        super().build(input_shape)

    def mask_softmax(self, x, mask):
        x_expe = tf.math.exp(x)
        x_expe_masked = x_expe * mask
        x_expe_sum = tf.reduce_sum(x_expe_masked, axis=-1)
        x_expe_sum = tf.expand_dims(x_expe_sum, axis=-1)
        softmax = x_expe_masked / x_expe_sum
        return softmax

    def call(self, x, mask=None):

        in_query, in_key, in_value = x

        Q = self.query_layer(in_query)
        K = self.key_layer(in_key)
        V = self.value_layer(in_value)

        batch_size = tf.shape(Q)[0]
        Q_seq_len = tf.shape(Q)[1]
        K_seq_len = tf.shape(K)[1]
        V_seq_len = tf.shape(V)[1]

        Q = tf.reshape(Q, [batch_size, Q_seq_len, self.nb_head, self.head_dim])
        K = tf.reshape(K, [batch_size, K_seq_len, self.nb_head, self.head_dim])
        V = tf.reshape(V, [batch_size, V_seq_len, self.nb_head, self.head_dim])

        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        Q = tf.reshape(Q, [batch_size * self.nb_head, Q_seq_len, self.head_dim])
        K = tf.reshape(K, [batch_size * self.nb_head, K_seq_len, self.head_dim])
        V = tf.reshape(V, [batch_size * self.nb_head, V_seq_len, self.head_dim])

        #Scaled dot product attention
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(256.)

        #MASK

        if mask is not None:
            QK = QK * mask
            softmax_QK = self.mask_softmax(QK, mask)
        else:
            softmax_QK = tf.nn.softmax(QK, axis=-1)

        attention = tf.matmul(softmax_QK, V)

        attention = tf.reshape(attention, [batch_size, self.nb_head, Q_seq_len, self.head_dim])
        attention = tf.transpose(attention, [0, 2, 1, 3])
        #Concat Multi Head
        attention = tf.reshape(attention, [batch_size, Q_seq_len, self.nb_head * self.head_dim])
        out_attention = self.out_proj(attention)
        return out_attention

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense_out = tf.keras.layers.Dense(256)
        super().build(input_shape)

    def call(self, x):
        attention = self.multi_head_attention((x, x, x))
        post_attention = self.norm(attention + x)
        x = self.dense_out(post_attention)
        enc_output = self.norm(x + post_attention)
        return enc_output

class Encoder(tf.keras.layers.Layer):

    def __init__(self, nb_encoder, **kwargs):
        self.nb_encoder = nb_encoder
        super(**kwargs).__init__()

    def build(self, input_shape):

        self.encoder_layers = []

        for nb in range(self.nb_encoder):
            self.encoder_layers.append(EncoderLayer())

        super().build(input_shape)

    def call(self, x):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.multi_head_self_attention = MultiHeadAttention()
        #self.multi_head_enc_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.proj_output = tf.keras.layers.Dense(256)

        super().build(input_shape)

    def call(self, x):
        enc_output, output_embedding, mask = x
        self_attention = self.multi_head_self_attention((output_embedding, output_embedding, output_embedding), mask=mask)
        post_self_attention = self.norm(output_embedding + self_attention)
        enc_attention = self.multi_head_self_attention((post_self_attention, enc_output, enc_output))
        post_enc_attention = self.norm(enc_attention + post_self_attention)
        proj_out = self.proj_output(post_enc_attention)
        dec_output = self.norm(proj_out + post_enc_attention)

        return dec_output

class Decoder(tf.keras.layers.Layer):

    def __init__(self, nb_decoder, **kwargs):
        self.nb_decoder = nb_decoder
        super(**kwargs).__init__()

    def build(self, input_shape):

        self.decoder_layers = []

        for nb in range(self.nb_decoder):
            self.decoder_layers.append(DecoderLayer())

        super().build(input_shape)

    def call(self, x):

        enc_output, output_embedding, mask = x

        dec_output = output_embedding
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer((enc_output, dec_output, mask))
        return dec_output

def get_transformer_model(output_voc):
    input_token = tf.keras.Input(shape=(5))
    output_token = tf.keras.Input(shape=(6))
 #Positional encoding
    input_pos_encoding = EmbeddingLayer(nb_token=5)(tf.range(5))
    output_pos_encoding = EmbeddingLayer(nb_token=6)(tf.range(6))

    #Retrieve embedding

    input_embedding = EmbeddingLayer(nb_token=5)(input_token)
    output_embedding = EmbeddingLayer(nb_token=6)(output_token)

    #Add the positional encoding
    input_embedding =  input_embedding + input_pos_encoding
    output_embedding = output_embedding + output_pos_encoding

    #Encoder
    enc_output = Encoder(nb_encoder=6)(input_embedding)

    #mask + decoder
    mask  = tf.sequence_mask(tf.range(6) + 1, 6)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=0)

    dec_output = Decoder(nb_decoder=6)((enc_output, output_embedding, mask))

    #Predictions
    out_pred = tf.keras.layers.Dense(len(output_voc))(dec_output)
    predictions = tf.nn.softmax(out_pred, axis=-1)

    model = tf.keras.Model([input_token, output_token], predictions)
    model.summary()
    return model

# transformer = get_transformer_model(output_voc)
# out = transformer((input_seq, output_seq))
# print (out.shape)
