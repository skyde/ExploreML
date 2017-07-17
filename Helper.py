import os
import tensorflow as tf

def validate_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def selu(x):
    with tf.variable_scope('selu') as _:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
