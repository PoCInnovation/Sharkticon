import tensorflow as tf
from Tokenizer import PacketTokenizer
from Transformer.transformer import Transformer
from Transformer.hyperparams import num_heads, d_model, dropout_rate, dff, num_layers
from Transformer.optimization import learning_rate, loss_function, train_loss, train_accuracy, accuracy_function
from Transformer.masks import create_masks
# from Transformer.masks import

BUFFER_SIZE = 20000
BATCH_SIZE = 64
PATH_VOCAB = "../data/vocab.txt"
PATH_DATASET = '../data/Dataset_test.csv'

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class SharkticonModel():
    def __init__(self):
        self.dataset = tf.data.experimental.CsvDataset(PATH_DATASET,
                                                       [tf.string, tf.string])
        train_packet = self.dataset.map(lambda packet, target: target)
        train_target = self.dataset.map(lambda packet, target: packet)
        self.reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]"]
        self.vocab_path = PATH_VOCAB
        self.tokenizer = PacketTokenizer(self.reserved_tokens, self.vocab_path)
        self.train_batches = self.make_batches(self.dataset)
        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=self.tokenizer.get_vocab_size(),
            target_vocab_size=self.tokenizer.get_vocab_size(),
            pe_input=1000,
            pe_target=1000,
            rate=dropout_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)

    def tokenize_pairs(self, packet, next_packet):
        packet = self.tokenizer.tokenize(packet)
        # Convert from ragged to dense, padding with zeros.
        packet = packet.to_tensor()

        next_packet = self.tokenizer.tokenize(next_packet)
        # Convert from ragged to dense, padding with zeros.
        next_packet = next_packet.to_tensor()
        return packet, next_packet

    def make_batches(self, ds):
        return (
            ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(self.tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
