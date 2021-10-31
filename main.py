from deepspeech import build_deepspeech_model
from preprocessing import build_dataset
from utils import CallbackEval

if __name__ == '__main__':

    # Add option to load model from file with argument in command line
    # Add number of epoch for training from command line

    epochs = 5
    train_dataset, validation_dataset = build_dataset(batch_size=32)
    model = build_deepspeech_model(rnn_units=512)
    callbacks = [CallbackEval(validation_dataset)]
    model.fit(
        train_dataset,
        validation_dataset=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    model.save("deepspeech.hdf5")


