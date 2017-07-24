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

def log10(x):
    num = tf.log(x)
    den = tf.log(tf.constant(10, dtype=num.dtype))
    return (tf.div(num, den))

def load_audio(path, force_mono=True):
    sample_rate, audio = wav.read(path)
    audio = audio.astype(np.float32)
    audio /= 32768

    if force_mono and len(audio.shape) > 1:
        audio = audio[:, 0:1]
    else:
        audio = np.expand_dims(audio, axis=1)

    return sample_rate, audio

def save_audio(audio, path, sample_rate):
    audio *= 32768
    audio = audio.astype(np.int16)
    wav.write(path, sample_rate, audio)
