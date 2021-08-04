# -*- coding: utf-8 -*-

from SharkticonModel import SharkticonModel
from Transformer.masks import create_masks

import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np
import logging

checkpoint_path = "./checkpoints/train"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def evaluate(request, max_length=1000):
        # inp request is portuguese, hence adding the start and end token
        request = tf.convert_to_tensor([request])
        request = sharkticon.tokenizer.tokenize(request).to_tensor()

        encoder_input = request

        start, end = sharkticon.tokenizer.tokenize([''])[0]
        output = tf.convert_to_tensor([start])
        output = tf.expand_dims(output, 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            predictions, attention_weights = sharkticon.transformer(encoder_input,
                                                                    output,
                                                                    False,
                                                                    enc_padding_mask,
                                                                    combined_mask,
                                                                    dec_padding_mask)

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)
            output = tf.concat([output, predicted_id], axis=-1)
            print(i, predicted_id, end)
            # return the result if the predicted_id is equal to the end token
            if predicted_id == end:
                print("OUR PACKET: ", request)
                break

        request_affichage = sharkticon.tokenizer.detokenize(output)[0]  # shape: ()
        tokens = sharkticon.tokenizer.lookup(output)[0]

        return request_affichage, tokens, attention_weights

def print_translation(request, tokens, ground_truth):
    print(f'{"Input:":15s}: {request}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

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

def plot_attention_weights(request, translated_tokens, attention_heads):
        in_tokens = tf.convert_to_tensor([request])
        in_tokens = sharkticon.tokenizer.tokenize(in_tokens).to_tensor()
        in_tokens = sharkticon.tokenizer.lookup(in_tokens)[0]
        in_tokens

        fig = plt.figure(figsize=(16, 8))

        for h, head in enumerate(attention_heads):
            ax = fig.add_subplot(2, 4, h + 1)

            plot_attention_head(in_tokens, translated_tokens, head)

            ax.set_xlabel(f'Head {h+1}')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sharkticon = SharkticonModel()

    ckpt_manager = tf.train.CheckpointManager(
        sharkticon.ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        sharkticon.ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    request = "GET[SEP]http://localhost:8080/tienda1/publico/anadir.jsp[SEP]Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)[SEP]en[SEP]close[SEP]null[SEP]null[SEP]JSESSIONID=B92A8B48B9008CD29F622A994E0F650D[SEP]"
    next_request = ""

    request_prediction, translated_tokens, attention_weights = evaluate(request)
    print_translation(request, request_prediction, next_request)
    #sharkticon.transformer.save("./saved_model")


    # head = 0
    # # shape: (batch=1, num_heads, seq_len_q, seq_len_k)
    # attention_heads = tf.squeeze(
    #     attention_weights['decoder_layer4_block2'], 0)
    # attention = attention_heads[head]
    # attention.shape

    # in_tokens = tf.convert_to_tensor([request])
    # in_tokens = sharkticon.tokenizer.tokenize(in_tokens).to_tensor()
    # in_tokens = sharkticon.tokenizer.lookup(in_tokens)[0]
    # in_tokens

    # translated_tokens

    # plot_attention_head(in_tokens, translated_tokens, attention)



    # plot_attention_weights(request, translated_tokens,
    #                     attention_weights['decoder_layer4_block2'][0])

    # """The model does okay on unfamiliar words. Neither "triceratops" or "encyclopedia" are in the input dataset and the model almost learns to transliterate them, even without a shared vocabulary:"""

    # request = "GET[SEP]http://localhost:8080/asf-logo-wide.gif~[SEP]HTTP/1.1[SEP]Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)[SEP]no-cache[SEP]no-cache[SEP]text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5[SEP]x-gzip, x-deflate, gzip, deflate[SEP]utf-8, utf-8;q=0.5, *;q=0.5[SEP]en[SEP]localhost:8080[SEP]close[SEP]null[SEP]null[SEP]JSESSIONID=51A7470173188BBB993947F2283059E4[SEP][SEP]anom[SEP]"
    # next_request = "http://localhost:8080/asf-logo-wide.gif~[SEP]HTTP/1.1[SEP]Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)[SEP]no-cache[SEP]no-cache[SEP]text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5[SEP]x-gzip, x-deflate, gzip, deflate[SEP]utf-8, utf-8;q=0.5, *;q=0.5[SEP]en[SEP]localhost:8080[SEP]close[SEP]null[SEP]null[SEP]JSESSIONID=51A7470173188BBB993947F2283059E4[SEP][SEP]anom[SEP]"

    # translated_text, translated_tokens, attention_weights = evaluate(request)
    # print_translation(request, translated_text, next_request)

    # plot_attention_weights(request, translated_tokens,
    #                     attention_weights['decoder_layer4_block2'][0])
