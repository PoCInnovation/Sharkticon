import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print("X before masking: \n", x)
    x = create_padding_mask(x)
    print("X after masking: \n", x)
    x = tf.random.uniform((1, 3))
    print("X before look-ahead mask :", x)
    temp = create_look_ahead_mask(x.shape[1])
    print("X after look-ahead mask :", temp)
