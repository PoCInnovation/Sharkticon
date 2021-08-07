import tensorflow as tf
import time

try:
    from SharkticonModel import SharkticonModel
    from Transformer.masks import create_masks
    from Transformer.optimization import loss_function, accuracy_function
    from Transformer.optimization import train_loss, train_accuracy
except Exception as e:
    from Model.SharkticonModel import SharkticonModel
    from Model.Transformer.masks import create_masks
    from Model.Transformer.optimization import loss_function, accuracy_function
    from Model.Transformer.optimization import train_loss, train_accuracy


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


def train(checkpoint_path, dataset_path):
    sharkticon = SharkticonModel(dataset_path)

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = sharkticon.transformer(inp, tar_inp,
                                                    True,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(
            loss, sharkticon.transformer.trainable_variables)
        sharkticon.optimizer.apply_gradients(
            zip(gradients, sharkticon.transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    ckpt_manager = tf.train.CheckpointManager(
        sharkticon.ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        sharkticon.ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    EPOCHS = 10
    BUFFER_SIZE = 1000
    BATCH_SIZE = 64

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(sharkticon.train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(
            f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


# if __name__ == '__main__':
    # train("./checkpoints/train")
