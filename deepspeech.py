from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils import ctc_loss
from preprocessing import CHARACTERS

FFT_LENGTH = 384

def build_deepspeech_model(input_dim: int = FFT_LENGTH // 2 + 1, output_dim: int = 31, rnn_layers: int = 5, rnn_units: int = 128):
    """ Build deepspeech 2 model

    Args:
        input_dim:
        output_dim:
        rnn_layers:
        rnn_units:
    """
    input_spectrogram = layers.Input((None, input_dim), name="input")
    #Expand dimension to use 2D CNN
    x = layers.Reshape((-1, input_dim, 1), name='expand_dim')(input_spectrogram)
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding='same',
        use_bias=False,
        name='conv_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation(activation='relu', name='relu_1')(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation(activation='relu', name='relu_2')(x)
    # Reshape tensor to feed the RNN layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(units=rnn_units*2, name='dense_1')(x)
    x = layers.ReLU(name='dense_1_relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    model = Model(inputs=input_spectrogram, outputs=output, name='deepspeech_2')

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=ctc_loss)

    return model