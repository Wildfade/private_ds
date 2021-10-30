from typing import Tuple
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DATA_PATH = keras.utils.get_file("LJSpeech-1.1", DATA_URL, untar=True)
WAVS_PATH = DATA_PATH + "/wavs/"
METADATA_PATH = DATA_PATH + "/metadata.csv"

CHARACTERS = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

def split_dataset(split_value: float = 0.90, frac : float = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Build dataset from downloaded data

    Args:
        split_value: volume of the dataset to be used for the training set
        frac: volume of the dataset to be kept to build the training and validation sets
    """
    metadata_df = pd.read_csv(METADATA_PATH, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    split = int(len(metadata_df) * split_value)

    return metadata_df[:split], metadata_df[split:]

def encode_single_sample(wav_file, label, frame_length: int = 256, frame_step: int = 160, fft_length: int = 384):
    """ Process and prepare audio file and labels for training

    Args:
        frame_length: Window length in samples
        frame_step: Number of samples to step
        fft_length: size of the fft to apply
        wav_file: audio file from the dataset
        label: label of the current sample corresponding to the audio file

    Returns:
        Spectrogram and transformerd label 
    """

    # Convert audio file to spectrogram
    file = tf.io.read_file(WAVS_PATH + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis = -1)
    audio = tf.cast(audio, dtype=tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.sqrt(spectrogram)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stdev = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stdev + 1e-10)

    # Process label
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding='UTF-8')
    label = keras.layers.StringLookup(vocabulary=CHARACTERS, oov_token="")(label)

    return spectrogram, label

def build_dataset(batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Build dataset

    Args:
        batch_size: Size of batches in the train and val dataset

    Returns:
        Train and val dataset
    """
    train_df, val_df = split_dataset(split_value=0.9, frac=1)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(train_df["file_name"]), list(train_df["normalized_transcription"]))
    )
    train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(val_df["file_name"]), list(val_df["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return train_dataset, validation_dataset





