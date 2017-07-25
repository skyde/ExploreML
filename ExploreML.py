import numpy as np
import os
import scipy.io.wavfile as wav
from PIL import Image
import tensorflow as tf
import random
import colorsys
import Helper

train_batch_size = 128
test_batch_size = 2
number_generator_filters = 32

max_epochs = 1000000

data_raw = "data_raw"
data_contains_name = None
data_train = "data_train"
data_train_reference = "data_train_reference"
data_test = "data_test"

save_dir = "save"
preview_dir = "preview"
train_from_scratch = False
save_progress = True

save_interval = 100
preview_interval = 10

encode_power = 5
encode_scalar = 0.03
fft_size = 256
source_size = int(fft_size / 2)
generate_size = source_size

step_size = int(source_size)

process_data_enabled = False
train_enabled = True

GAN_output_size = 32
epsilon = 1e-12
learning_rate = 0.0002
learning_rate_L1 = 100
learning_rate_GAN = 1
beta1 = 0.5

encode_fft_reuse = False
def encode_fft(audio, size=512, steps=512, step_offset=0):

    return_values = []

    with tf.variable_scope("encode_fft", reuse=encode_fft_reuse):
        input = tf.placeholder(tf.float32, [size], "input")
        output = tf.fft(tf.cast(input, tf.complex64))
        output = output[:int(size / 2)]

        output = output[:int(source_size)]

        sess = tf.get_default_session()

        for i in range(steps):
            offset = step_offset + i * size
            feed_audio = audio[offset: offset + size, 0]

            values = sess.run({"output": output}, feed_dict={input: feed_audio})
            output_values = values["output"]

            output_values = np.expand_dims(output_values, axis=0)

            return_values.append(output_values)

    global encode_fft_reuse
    encode_fft_reuse = True

    return np.concatenate(return_values, axis=0)

fft_to_audio_init = False
def fft_to_audio(fft):

    return_values = []
    with tf.variable_scope("fft_to_audio", reuse=fft_to_audio_init):
        input = tf.placeholder(tf.complex64, [fft.shape[1]])

        zeroes = tf.fill([int(fft_size / 2) - int(input.shape[0])], input[-1] * 0)
        fill = tf.concat([input, zeroes], axis=0)

        inverse = tf.reverse(tf.conj(fill), [0])
        full = tf.concat([fill, inverse[-2: -1], inverse[:-1]], axis=-1)
        output = tf.cast(tf.ifft(full), tf.float32)

        sess = tf.get_default_session()

        steps = fft.shape[0]
        for i in range(steps):
            feed_audio = fft[i, :]

            values = sess.run({"output": output}, feed_dict={input: feed_audio})
            output_values = values["output"]

            return_values.append(output_values)

    global fft_to_audio_init
    fft_to_audio_init = True

    return np.concatenate(return_values, axis=0)

def save_fft_to_image(fft, path):
    real = np.expand_dims(fft.real, axis=-1)
    imag = np.expand_dims(fft.imag, axis=-1)
    zeros = np.zeros(real.shape)
    image_values = np.concatenate([real, imag, zeros], axis=-1)

    image_values = image_values * encode_scalar
    image_values = np.power(np.abs(image_values), 1 / encode_power) * np.sign(image_values)

    image_values /= 2
    image_values += 0.5
    print("max " + str(np.max(image_values)))
    print("min " + str(np.min(image_values)))

    image_values *= 255

    image_values = image_values.astype('uint8')
    image = Image.fromarray(image_values)
    image.save(path)

def load_fft_from_image(path):
    with open(path, 'rb') as p:
        image = Image.open(p)
        values = np.asarray(image, dtype="float32")
        del image

    values /= 255
    values -= 0.5
    values *= 2
    values = np.power(np.abs(values), encode_power) * np.sign(values)
    values = values / encode_scalar

    complex_values = np.array(values[:, :, 0], dtype=complex)
    complex_values.imag = values[:, :, 1]

    return complex_values

def save_image_as_audio(fft, path, sample_rate):
    audio = fft_to_audio(fft)
    Helper.save_audio(audio, path, sample_rate)

def process_data():
    audio_index = 0
    i = 0
    with tf.Session() as sess:
        for dir_name, _, file_list in os.walk(data_raw):
            if data_contains_name is None or data_contains_name in os.path.normpath(dir_name):
                for file_name in file_list:
                    path = dir_name + "\\" + file_name
                    # path = path.replace('\\', '/')

                    # print(file_name)

                    sample_rate, audio = Helper.load_audio(path)

                    step_index = 0
                    # print(audio.shape[0])

                    for step_offset in range(0, audio.shape[0] - fft_size * fft_size, fft_size * fft_size):
                        output_name = str(audio_index) + "-" + str(step_index)

                        folder = data_train

                        if i % 5 == 0:
                            folder = data_test

                        # print(output_name)
                        # print(step_offset)

                        # Image
                        fft = encode_fft(audio, size=fft_size, steps=fft_size, step_offset=step_offset)
                        image_output_path = folder + "\\" + output_name + ".png"
                        Helper.validate_directory(folder)
                        save_fft_to_image(fft, image_output_path)

                        # Reference Audio
                        complex_values = load_fft_from_image(image_output_path)
                        Helper.validate_directory(data_train_reference)
                        audio_output_path = data_train_reference + "\\" + output_name + ".wav"
                        save_image_as_audio(complex_values, audio_output_path, sample_rate)
                        # output_audio = fft_to_audio(complex_values)
                        # Helper.save_audio(output_audio, audio_output_path, sample_rate)

                        step_index += 1
                        i += 1

                    audio_index += 1

def conv(current, out_channels, apply_activation=True):
    # current = batch_input

    with tf.variable_scope("conv"):
        in_channels = current.get_shape()[3]
        filter = tf.get_variable("filter",
                                 [4, 4, in_channels, out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        current = tf.nn.conv2d(current, filter, [1, 2, 2, 1], padding="SAME")

        w = tf.get_variable("encoder_bias",
                            [1, 1, 1, out_channels],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))

        if apply_activation:
            current = Helper.selu(current + w)

    return current

def deconv(current, out_channels):
    out_channels = int(out_channels)

    with tf.variable_scope("deconv"):
        shape = current.get_shape()
        in_channels = shape[3]
        filter = tf.get_variable("filter",
                                 [4, 4, out_channels, in_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        dyn_input_shape = tf.shape(current)[0]
        local_batch = dyn_input_shape
        output_shape = tf.stack([local_batch,
                                 current.shape[1] * 2,
                                 current.shape[2] * 2,
                                 out_channels])

        current = tf.nn.conv2d_transpose(current, filter, output_shape, [1, 2, 2, 1], padding="SAME")
        current = tf.reshape(current, output_shape)

        w = tf.get_variable("decoder_bias",
                            [1, 1, 1, out_channels],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))

        current = Helper.selu(current + w)

    return current

def caculate_layer_depth(size):
    if size == source_size / 2:
        return 2 * number_generator_filters
    elif size == source_size / 4:
        return 4 * number_generator_filters
    # elif size == 2:
    #     return 12 * number_generator_filters
    # elif size == 1:
    #     return 16 * number_generator_filters

    return 8 * number_generator_filters

def create_generator(current, dropout=True, output_size=1, output_channels=None, output_activated=True):
    print("generator input " + str(current.shape))

    layers = []

    layer_index = 0

    layers.append(current)

    while current.shape[1] > output_size:
        with tf.variable_scope("conv_" + str(layer_index)):
            next_size = int(current.shape[1]) / 2
            depth = caculate_layer_depth(next_size)

            apply_activation = True
            if next_size == output_size:
                if not output_activated:
                    output_activated = False

                if output_channels is not None:
                    depth = output_channels

            current = conv(current, depth, apply_activation=apply_activation)

            print("generator layer " + str(current.shape))

            if dropout and current.shape[1] <= 32:
                current = tf.nn.dropout(current, 0.4)

            layers.append(current)

            layer_index += 1

    return current, layers

def create_decoder(current, layers, dropout=False, output_channels=2):
    print("decoder input " + str(current.shape))

    layer_index = 0
    while current.shape[1] < source_size:
        with tf.variable_scope("deconv_" + str(layer_index)):
            depth = caculate_layer_depth(int(current.shape[1]) * 2)

            if current.shape[1] * 2 == source_size:
                depth = output_channels

            for layer in layers:
                if layer.shape[1] == current.shape[1]:
                    current = tf.concat([current, layer], axis=3)
                    print("skip " + str(current.shape))

            current = deconv(current, depth)

            print("decoder layer " + str(current.shape))

            if dropout and current.shape[1] <= 32:
                current = tf.nn.dropout(current, 0.4)

            layer_index += 1

    return current

def output_to_rgb(image):
    image = image * 0.5 + 0.5
    return tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, 1]])

def create_chain(feed):

    list_generated = []
    list_truth = []
    list_predict_real = []
    list_predict_fake = []

    current = feed[:, :source_size, :, :]
    source = current

    reuse = False
    for offset in range(0, feed.shape[1] - source_size, step_size):
        truth = feed[:, step_size + offset: step_size + offset + generate_size, :, :]

        # if offset != 0:
        #     left = source[:, step_size:, :, :]
        #     right = current[:, -step_size:, :, :]
        #     print("left.shape " + str(left.shape))
        #     print("right.shape " + str(right.shape))
        #     current = tf.concat([left, right], 1)

        # source = current

        with tf.variable_scope("generator", reuse=reuse):
            generator_input = tf.concat([current, source], axis=3)
            current, layers = create_generator(generator_input)
            print("generator " + str(current))
            current = create_decoder(current, layers)
            print("decoder " + str(current))

        # current = tf.reshape(current, [ int(feed.shape[0]),
        #                                 int(current.shape[1]),
        #                                 int(current.shape[2]),
        #                                 int(current.shape[3])])
        generated = current

        print("generated " + str(generated.shape))
        print("truth " + str(truth.shape))

        # GAN
        real_input = tf.concat([source, truth], axis=3)
        fake_input = tf.concat([source, generated], axis=3)

        print("discriminator input shape " + str(real_input.shape))

        with tf.variable_scope("discriminator", reuse=reuse):
            predict_real, _ = create_generator(real_input, output_size=GAN_output_size, dropout=True,
                                      output_channels=1, output_activated=False)

        with tf.variable_scope("discriminator", reuse=True):
            predict_fake, _ = create_generator(fake_input, output_size=GAN_output_size, dropout=True,
                                      output_channels=1, output_activated=False)

            predict_real = tf.sigmoid(predict_real)
            predict_fake = tf.sigmoid(predict_fake)

        reuse = True

        list_generated.append(generated)
        list_truth.append(truth)
        list_predict_real.append(predict_real)
        list_predict_fake.append(predict_fake)

    combined_generated = tf.concat(list_generated, axis=0)
    combined_truth = tf.concat(list_truth, axis=0)
    combined_predict_real = tf.concat(list_predict_real, axis=0)
    combined_predict_fake = tf.concat(list_predict_fake, axis=0)

    return combined_generated, combined_truth, combined_predict_real, combined_predict_fake
        # return current

        # with tf.variable_scope("generator", reuse=True):
        #     preview_current, preview_layers = create_generator(source)
        #     print("preview generator " + str(preview_current))
        #     preview_current = create_decoder(preview_current, preview_layers)
        #     print("preview decoder " + str(preview_current))

# def get_image_paths(directory_path):
#     paths = []
#     for dir_name, _, file_list in os.walk(directory_path):
#         for file_name in file_list:
#             path = dir_name + "\\" + file_name
#             path = path.replace('\\', '/')
#
#             paths.append(path)
#
#     return paths

# def load_image(path):
#     with open(path, 'rb') as p:
#         image = Image.open(p)
#         data = np.asarray(image, dtype="float32")
#         del image
#
#     data /= 255
#     data = data * 2 - 1.0
#     data = data[:, :, :2]
#
#     return data

def save_preview(sess, step, state, source_rgb, generated_rgb, predict_real, predict_fake, truth_rgb, summary, writer):

    # for path in get_image_paths(data_test):
        # image = load_image(path)

        # for offset in range(0, image.shape[0] - generate_size - source_size, generate_size):
        #     feed_image = image[offset: offset + source_size + generate_size]
        #     feed_image = np.expand_dims(feed_image, axis=0)

    run = sess.run({
        "source": source_rgb,
        "generated": generated_rgb,
        "predict_real": predict_real,
        "predict_fake": predict_fake,
        "truth": truth_rgb,
        "summary": summary
    }, feed_dict={state: "test"})

    preview_path = preview_dir + "\\" + "step_" + str(step)
    generated_path = preview_path + "_generated.png"
    Helper.validate_directory(preview_dir)
    Helper.save_image(run["source"][0, :, :, :], preview_path + "_source.png")
    Helper.save_image(run["generated"][-1, :, :, :], generated_path)
    Helper.save_image(run["truth"][-1, :, :, :], preview_path + "_truth.png")
    Helper.save_image(run["predict_real"][-1, :, :, :], preview_path + "_predict_real.png")
    Helper.save_image(run["predict_fake"][-1, :, :, :], preview_path + "_predict_fake.png")

    fft = load_fft_from_image(generated_path)
    # print(fft.shape)
    save_image_as_audio(fft, preview_path + "_audio.wav", 22000)

    writer.add_summary(run["summary"], step)

def train():
    state = tf.placeholder(tf.string)

    train_batch = Helper.image_batch(data_train, source_size + generate_size, source_size, train_batch_size, channels=2)
    test_batch = Helper.image_batch(data_test, source_size + generate_size, source_size, test_batch_size, channels=2)

    feed = tf.cond(tf.equal(state, "train"), lambda: train_batch, lambda: test_batch)

    generated, truth, predict_real, predict_fake = create_chain(feed)

    with tf.name_scope("discriminator_loss"):
        discriminator_loss = tf.reduce_mean(-(tf.log(predict_real + epsilon) + tf.log(1 - predict_fake + epsilon)))
        discriminator_loss = discriminator_loss * learning_rate_GAN

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + epsilon))
        gen_loss_L1 = tf.reduce_mean(tf.abs(generated - truth))

        generator_loss = gen_loss_GAN * learning_rate_GAN + gen_loss_L1 * learning_rate_L1

    with tf.name_scope("discriminator_train"):
        disc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

        disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
        disc_grads_and_vars = disc_optimizer.compute_gradients(discriminator_loss, var_list=disc_tvars)
        discriminator_train = disc_optimizer.apply_gradients(disc_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discriminator_train]):
            tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
            gen_grads_and_vars = gen_optimizer.compute_gradients(generator_loss, var_list=tvars)
            generator_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    train_group = tf.group(generator_train, incr_global_step)

    # Output
    source = feed[:, :source_size, :, :]
    source_rgb = output_to_rgb(source)
    generated_rgb = output_to_rgb(generated)
    truth_rgb = output_to_rgb(truth)

    with tf.Session() as sess:
        with tf.name_scope("summary"):
            tf.summary.image("source", source_rgb)
            tf.summary.image("final_generated", generated_rgb[-2:-1, :, :, :])
            tf.summary.image("generated", generated_rgb)
            tf.summary.image("truth", truth_rgb)
            tf.summary.image("predict_real", predict_real)
            tf.summary.image("predict_fake", predict_fake)
            tf.summary.scalar("gen_loss_L1", gen_loss_L1)
            tf.summary.scalar("gen_loss_GAN", gen_loss_GAN)
            tf.summary.scalar("discriminator_loss", discriminator_loss)

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(save_dir, graph=sess.graph, max_queue=10,
               flush_secs=120)
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint is not None and not train_from_scratch:
            print("Loading checkpoint " + latest_checkpoint)
            saver.restore(sess, latest_checkpoint)

        step = global_step.eval()

        while step < max_epochs:
            run = sess.run({
                "train": train_group,
                "global_step": global_step
            }, feed_dict={state: "train"})

            step = run["global_step"]

            print("step " + str(step))

            if step % preview_interval == 0:
                save_preview(sess, step, state, source_rgb, generated_rgb, predict_real, predict_fake, truth_rgb, summary, writer)

            if step % save_interval == 0:
                Helper.validate_directory(save_dir)
                if save_progress:
                    _ = saver.save(sess, save_dir + "/model.ckpt", global_step=global_step)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if process_data_enabled:
    process_data()

if train_enabled:
    train()
