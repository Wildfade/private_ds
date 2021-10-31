from deepspeech import build_deepspeech_model
from preprocessing import build_dataset
from utils import CallbackEval


if __name__ == '__main__':
    
    model = build_deepspeech_model(output_dim=31, rnn_units=512)
    model.summary()
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


