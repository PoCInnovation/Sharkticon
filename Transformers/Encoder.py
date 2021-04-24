"""
1) Chaque EncoderLayer se compose de sous-couches:

    -> MHA (avec padding mask)
    -> Point wise feed forward networks

    Chacune de ces sous-couches a une connexion résiduelle autour d'elle, suivie d'une normalisation de layer.
    Les connexions résiduelles aident à éviter le problème de vanishGradient dans les réseaux profonds.

    La sortie de chaque sous-layer est LayerNorm(x + Sublayer(x)).
    La normalisation se fait sur le d_model (dernier) axe.
    Il y a N encoder layers dans le transformer.



2)L'Encodeur c'est:

    Input Embedding
    Positional Encoding
    N EncoderLayers

The input est soumise à l'embedding qui est additionné (vecteur) au positional encoding
La sortie de cette somme est l'input des EncoderLayers
La sortie de l'encodeur est l'entrée du Decodeur




"""

from Mha import MultiHeadAttention, tf, point_wise_feed_forward_network
from encoding import positional_encoding

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nb_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, nb_heads)
        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attention_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(x + attention_output)  # (batch_size, input_seq_len, d_model)

        feed_forward_output = self.feed_forward(out1)  # (batch_size, input_seq_len, d_model)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        out2 = self.layernorm2(out1 + feed_forward_output)  # (batch_size, input_seq_len, d_model)

        return out2

# TODO:adapt to Network dataset
class Encoder(tf.keras.layers.Layer):
    def __init__(self, hyper_params, maximum_position_encoding):
        super(Encoder, self).__init__()

        self.d_model = hyper_params["d_model"]
        self.rate = hyper_params["rate"]
        self.nb_layers = hyper_params["nb_encoder_layers"]
        self.embedding = tf.keras.layers.Embedding(hyper_params["input_size"], self.d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model, hyper_params["nb_heads"],
                           hyper_params["dff"], self.rate) for _ in range(self.nb_layers)]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.nb_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


if __name__ == "__main__":
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

    hyper_params = {
        "num_layers": 4,
        "d_model": 128,
        "dff": 512,
        "num_heads": 8,
        "rate": 0.1,
        "input_size": 8500,
        "target_size": 8000,
        "nb_encoder_layers": 2
    }

    sample_encoder = Encoder(hyper_params, max_position_encoding=10000)

    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
