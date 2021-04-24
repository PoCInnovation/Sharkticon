import tensorflow as tf
import numpy as np

''' ! Attention(q, k, v) = softmax_k_normalized * ((Q*K / sqrt(dk) ** mask) * v
    dk = variance de QK
'''

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def print_sdp_attention(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)
    print("_________________________")

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                         [0, 10, 0],
                         [0, 0, 10],
                         [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                         [10, 0],
                         [100, 5],
                         [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_sdp_attention(temp_q, temp_k, temp_v)

    # This query aligns with a repeated key (third and fourth),
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_sdp_attention(temp_q, temp_k, temp_v)

    # This query aligns equally with the first and second key,
    # so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_sdp_attention(temp_q, temp_k, temp_v)
