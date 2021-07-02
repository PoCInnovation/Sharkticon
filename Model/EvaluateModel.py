from ModelTrainer import create_masks, tf, tokenizers, transformer
import matplotlib as plt


def evaluate(packet, max_length=40):
    # inp sentence is portuguese, hence adding the start and end token
    packet = tf.convert_to_tensor([packet])
    packet = tokenizers.tokenize(packet).to_tensor()

    encoder_input = packet

    start, end = tokenizers.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.argmax(predictions, axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break

    packet_affichage = tokenizers.detokenize(output)[0]  # shape: ()
    tokens = tokenizers.lookup(output)[0]

    return packet_affichage, tokens, attention_weights


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


packet = "178.16.0.5[SEP]58445[SEP]192.168.50.1[SEP]4463[SEP]"
next_packet = "172.16.0.5[SEP]36908[SEP]192.168.50.1[SEP]9914[SEP]"

packet_prediction, translated_tokens, attention_weights = evaluate(packet)
print_translation(packet, packet_prediction, next_packet)


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(
        labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


head = 0
# shape: (batch=1, num_heads, seq_len_q, seq_len_k)
attention_heads = tf.squeeze(
    attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]
print(attention.shape)
in_tokens = tf.convert_to_tensor([packet])
in_tokens = tokenizers.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.lookup(in_tokens)[0]
print(in_tokens)
print(translated_tokens)
plot_attention_head(in_tokens, translated_tokens, attention)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.lookup(in_tokens)[0]
    in_tokens

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h + 1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()

plot_attention_weights(packet, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])

packet = "Eu li sobre triceratops na enciclopédia."
ground_truth = "I read about triceratops in the encyclopedia."

translated_text, translated_tokens, attention_weights = evaluate(packet)
print_translation(packet, translated_text, ground_truth)

plot_attention_weights(packet, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])