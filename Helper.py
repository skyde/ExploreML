import numpy as np
import os
import tensorflow as tf
import scipy.io.wavfile as wav
from PIL import Image

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

def save_image(image, path):
    # if len(image.shape) == 4:
    #     image = image[0, :, :, :]
    image = image * 255.0

    image = image.astype('uint8')

    if image.shape[2] is 1:
        image = np.broadcast_to(image, (image.shape[0], image.shape[1], 3))

    image = Image.fromarray(image)

    # if resize_to != -1:
    #     image = image.resize((resize_to, resize_to), Image.BILINEAR)

    image.save(path)

def image_batch(path, size_y, size_x, batch_size, channels=3):
    file_names = tf.train.match_filenames_once(os.path.join(path, "*.*"))

    print(file_names)

    filename_queue = tf.train.string_input_producer(file_names, shuffle=False, seed=42)

    # print(filename_queue)

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    images = tf.image.decode_png(image_file)
    images.set_shape((size_y, size_x, 3))
    if channels != 3:
        images = images[:, :, 0:channels]
    images = tf.cast(images, tf.float32)
    images = images / 255.0
    image = images * 2.0 - 1.0

    return tf.train.batch([images], batch_size=batch_size)
