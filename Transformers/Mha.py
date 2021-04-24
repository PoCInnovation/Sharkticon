'''
Chaque bloc d'attention à plusieurs têtes reçoit trois entrées;
Q (requête), K (clé), V (valeur).
Ceux-ci sont placés à travers des couches linéaires (Dense) et divisés en plusieurs têtes.

Le scaled_dot_product_attention est appliqué à chaque tête
(diffusé par souci d'efficacité).
Un masque approprié doit être utilisé dans l'étape d'attention.
La sortie d'attention pour chaque tête est ensuite concaténée
(en utilisant tf.transpose, et tf.reshape) et subir un Dense Layer final.
'''

from Sdpa import scaled_dot_product_attention as sdpa
from modules import tf, np

# ! Attention au format de v, k, q, mask

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=512, nb_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.nb_heads = nb_heads
        self.d_model = d_model

        assert d_model % self.nb_heads == 0

        self.depth = d_model // self.nb_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (nb_heads, depth).
        Transpose the result such that the shape is (batch_size, nb_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.nb_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):

        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, nb_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, nb_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, nb_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, nb_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, nb_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdpa(q, k, v, mask)

        # ? concat
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, nb_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

if __name__ == "__main__":
    temp_mha = MultiHeadAttention(d_model=512, nb_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha.call(y, k=y, q=y, mask=None)
    print("Output Shape:", out.shape)
    print("Attention Shape:", attn.shape)
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print("Pt2pt feed forward:", sample_ffn(tf.random.uniform((64, 50, 512))).shape)
