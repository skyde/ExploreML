import numpy as np
import os
import tensorflow as tf
import scipy.io.wavfile as wav

def validate_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def selu(x):
    with tf.variable_scope('selu') as _:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def save_audio(filename, directory, audio, sample_rate, scalar=32768.0, copy=True):
    if copy:
        audio = np.copy(audio)
    audio *= scalar
    audio = audio.astype(np.int16)

    validate_directory(directory)
    wav.write(directory + "\\" + filename + ".wav", sample_rate, audio)