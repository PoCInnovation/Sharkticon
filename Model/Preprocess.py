import tensorflow as tf
import pathlib
import re
import tensorflow_text as text

BUFFER_SIZE = 1000
BATCH_SIZE = 64
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]"]
vocab_path = "./Vocab/my-tokenizer.json"


def create_dataset(path_to_csv, cols=[]):
    dataset = tf.data.experimental.CsvDataset(path_to_csv, cols)
    return dataset


def add_start_end(ragged):
    START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


class PacketTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

try:
    tokenizers = PacketTokenizer(reserved_tokens, vocab_path)
    dataset = create_dataset('./Datasets_created/Dataset_TOTO.csv', [tf.string, tf.string])
except Exception as e:
    print("Error : ", e)
    exit(0)


def tokenize_pairs(packet, next_packet):
    packet = tokenizers.tokenize(packet)
    # Convert from ragged to dense, padding with zeros.
    packet = packet.to_tensor()

    next_packet = tokenizers.tokenize(next_packet)
    # Convert from ragged to dense, padding with zeros.
    next_packet = next_packet.to_tensor()
    return packet, next_packet


def make_batches(ds):
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

train_batches = make_batches(dataset)
print(train_batches)

if __name__ == "__main__":
    for input_packet, target_packet in dataset.batch(3).take(1):
        print(input_packet)
        for pckt in input_packet.numpy():
            print(pckt.decode('utf-8'))
        print()
        for packet in target_packet.numpy():
            print(packet.decode('utf-8'))

        for packet in input_packet.numpy():
            print(packet.decode('utf-8'))

    encoded = tokenizers.tokenize(input_packet)
    print(encoded)
    for row in encoded.to_list():
        print(row)
