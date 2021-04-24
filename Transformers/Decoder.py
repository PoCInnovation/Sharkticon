'''
Chaque layer de décodeur se compose de hidden-layers:

    --> MHA masquée (avec look ahead masket et padding mask)
    --> MHA (avec padding mask).
        --> V (value) et K (clé) reçoivent la sortie de l'encodeur en tant qu'inputs.
        --> Q (query) reçoit la sortie de la hidden-layer masked MHA.
    --> Point wise feed forward networks


    Réseaux à réaction ponctuelle

    Chacun de ces hidden-layers a une connexion résiduelle autour d'elle suivie d'une normalisation de layer.
    La sortie de chaque hidden-layer est LayerNorm(x + Sublayer(x)).
    La normalisation se fait sur le d_model (dernier) axe.

    Il y a N Decoder layers dans le transformer.

    Lorsque Q reçoit la sortie du premier bloc d'attention du décodeur et que K reçoit la sortie du encodeur,
    les poids d'attention représentent l'importance accordée à l'entrée du décodeur sur la base de la sortie de l'encodeur.
    En d'autres termes, le décodeur prédit le mot suivant:
        - en regardant la sortie du encodeur
        - en s'occupant lui-même de sa propre sortie.
'''

from Mha import tf, np, MultiHeadAttention, point_wise_feed_forward_network
from Encoder import EncoderLayer, positional_encoding, Encoder

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    # TODO: simplify
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape ==> (batch_size, input_seq_len, d_model)

        attention1, attention1_weights_block = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(attention1 + x)

        attention2, attention2_weights_block = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(attention2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.feed_forward(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attention1_weights_block, attention2_weights_block

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        attention_weights[f'decoder_layer{i+1}_block1'] = block1
        attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

if __name__ == "__main__":
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    sample_encoder = Encoder(nb_layers=2, d_model=512, nb_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)

    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    # ? -----------------Partie Decoder--------------

    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_layer_output, block_weihgts, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)
    print(sample_decoder_layer_output.shape, block_weihgts.shape)  # (batch_size, target_seq_len, d_model)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             max_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output= sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)
