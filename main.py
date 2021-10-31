from deepspeech import build_deepspeech_model
from preprocessing import build_dataset
from utils import decode_batch_predictions, NUM_TO_CHAR
import tensorflow as tf
from tensorflow import keras
from jiwer import wer
import numpy as np

model = build_deepspeech_model(rnn_units=512)


class CallbackEval(keras.callbacks.Callback):
    """ Displays a batch of outputs after every epoch
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            x, y = batch
            batch_predictions = self.model.predict()
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

epochs = 5
train_dataset, validation_dataset = build_dataset(batch_size=32)
callbacks = [CallbackEval(validation_dataset)]
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=callbacks
)
model.save("deepspeech.hdf5")


