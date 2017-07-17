import numpy as np
import os
import scipy.io.wavfile as wav
from PIL import Image
import tensorflow as tf
import random
import colorsys
import Helper

# input_size = 128
# output_size = 128
batch_size = 512
max_preview_export = 4
number_generator_filters = 8

max_epochs = 40000

data_source = "data"
data_contains_name = "piano"
data_input = "data_input"

save_dir = "save"

process_window = 2**16

# def process_data():
#     for dir_name, _, file_list in os.walk(data_source):
#         if data_contains_name is None or data_contains_name in os.path.normpath(dir_name):
#             for file_name in file_list:
#                 path = dir_name + "\\" + file_name
#                 path = path.replace('\\', '/')
#
#                 sample_rate, audio = wav.read(path)
#                 Helper.validate_directory(data_input)
#                 audio = audio.astype(np.float64)
#                 audio /= 32768
#
#                 print(sample_rate)
#
#                 audio *= 32768
#                 audio = audio.astype(np.int16)
#                 wav.write(data_input + "\\" + file_name, sample_rate, audio)

# process_data()

#
#
# def lrelu(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)
#
#
def conv(batch_input, out_channels, stride=2, activation="selu"):
    current = batch_input

    in_channels = current.get_shape()[2]
    filter = tf.get_variable("filter", [4, in_channels, out_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_size, in_channels], [filter_size, in_channels, out_channels]
    #     => [batch, out_size, out_channels]
    current = tf.nn.conv1d(current, filter, stride, padding="SAME")

    w = tf.get_variable("encoder_bias", [1, 1, out_channels], dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0, 0.02))

    if activation == "selu":
        current = Helper.selu(current + w)

    return current
#
#
# def deconv(current, out_channels, reuse=False, activation="selu"):
#     out_channels = int(out_channels)
#
#     with tf.variable_scope("deconv"):
#         # batch_input = tf.reshape(batch_input, (int(batch_size),
#         #                                        int(batch_input.shape[1]),
#         #                                        int(batch_input.shape[2]),
#         #                                        int(batch_input.shape[3])))
#         # print("deconv input " + str(batch_input.shape))
#         # print("deconv output channels " + str(out_channels))
#
#
#         # current = batch_input
#         shape = current.get_shape()
#         in_channels = shape[3]
#         filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
#                                  initializer=tf.random_normal_initializer(0, 0.02))
#
#         # [256, 128, 128, 3]
#         dyn_input_shape = tf.shape(current)[0]
#         # print("dyn_input_shape " + str(dyn_input_shape))
#         local_batch = dyn_input_shape
#         # local_batch = tf.cond(tf.equal(shape[0], None), 8, shape[0])
#         output_shape = tf.stack([local_batch,
#                                  current.shape[1] * 2,
#                                  current.shape[2] * 2,
#                                  out_channels])
#
#         # print(output_shape)
#         current = tf.nn.conv2d_transpose(current, filter, output_shape, [1, 2, 2, 1], padding="SAME")
#
#         # current = add_upscale(batch_input)
#         # current, _ = conv(current, out_channels, reuse=reuse)
#         current = tf.reshape(current, output_shape)
#
#         # print("deconv output " + str(current.shape))
#         # tf.nn.conv
#
#         w = tf.get_variable("decoderBias", [1, 1, 1, out_channels], dtype=tf.float32,
#                             initializer=tf.random_normal_initializer(0, 0.02))
#         if activation == "selu":
#             current = selu(current + w)
#
#     return current
#
#
# def add_upscale(self):
#     """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""
#     prev_shape = self.get_shape()
#     size = [2 * int(s) for s in prev_shape[1:3]]
#     return tf.image.resize_nearest_neighbor(self, size)
#
# def encode_color(image, encoding="rgb"):
#     if encoding is "one_hot_h":
#         # rgb = image
#         hsv = tf.image.rgb_to_hsv(image)
#         sv = hsv[:, :, :,  1:3]
#         s = sv[:, :, :, 0:1]
#         s *= 1000
#         # Red, Green, Blue, Saturation, Value
#         # image = tf.concat((rgb, sv), axis=2)
#         h = hsv[:, :, :,  0]
#         h *= color_one_hot_dim
#         h = tf.cast(h, tf.int32)
#         one_hot = tf.one_hot(h, color_one_hot_dim, axis=3)
#         print("one hot " + str(one_hot.shape))
#
#         # image = tf.concat((sv, one_hot), axis=3)
#         image = tf.concat((sv, one_hot), axis=3)
#         print("image.shape " + str(image.shape))
#     # if color_space is "xysv":
#     else:
#         image = image * 2.0 - 1.0
#
#     return image
#
#
# def decode_color(image, encoding="rgb"):
#
#     if encoding is "one_hot_h":
#         # rgb = image[:, :, :, 0:3]
#         # rgb_to_hsv = tf.image.rgb_to_hsv(rgb)
#
#         one_hot = image[:, :, :, 2:color_one_hot_dim + 2]
#         sv = image[:, :, :, :2]
#
#         h = tf.arg_max(one_hot, 3) / color_one_hot_dim
#         h = tf.cast(h, tf.float32)
#         h = tf.expand_dims(h, axis=3)
#
#         print("h " + str(h.shape))
#
#         hsv = tf.concat((h, sv), axis=3)
#         # h = rgb_to_hsv[:, :, :, :1]
#         # sv = image[:, :, :, 3:5]
#         # hsv = tf.concat((h, sv), axis=3)
#
#         image = tf.image.hsv_to_rgb(hsv)
#     else:
#         image = (image * 0.5 + 0.5)
#
#     return image
#
#
# # def get_batch(directory, offset, size):
# #     images = []
# #     paths = os.listdir(directory)
# #
# #     # batch_size = 32
# #     current = 0
# #     for i in range(size):
# #         # print((i + offset) % len(paths))
# #         path = paths[(i + offset) % len(paths)]
# #         # print(path)
# #         image = Image.open(directory + "/" + path)
# #
# #         data = np.asarray(image, dtype="float32")
# #         data = data[:, :, :3]
# #         data = data / 255.0
# #         data = encode_color(data)
# #         data = data * 2 - 1.0
# #         data = np.expand_dims(data, axis=0)
# #
# #         images.append(data)
# #
# #         current += 1
# #
# #         if current >= size:
# #             break
# #         # print(current)
# #         # print(all_images)
# #
# #     return np.concatenate(images, axis=0)
#
# def layer_depth(size):
#     if size == 1:
#         return number_generator_filters * 8
#     elif size == 2:
#         return number_generator_filters * 8
#     elif size == 4:
#         return number_generator_filters * 8
#     elif size == 8:
#         return number_generator_filters * 8
#     elif size == 16:
#         return number_generator_filters * 8
#     elif size == 32:
#         return number_generator_filters * 4
#     elif size == 64:
#         return number_generator_filters * 2
#     elif size == 128:
#         return number_generator_filters * 2
#     elif size == 256:
#         return number_generator_filters * 2
#     elif size == 512:
#         return number_generator_filters * 2
#     elif size == 1024:
#         return number_generator_filters * 2
#     return number_generator_filters
#
#
# def encoder(current, reuse=False, max_size=1, dropout=False, output_channels=None, output_rectified=True):
#     layers = []
#     i = 0
#     # with tf.variable_scope(name, reuse=reuse):
#     while current.shape[1] > max_size:
#         with tf.variable_scope("encoder_" + str(i), reuse=reuse):
#             if dropout:
#                 current = tf.nn.dropout(current, 0.4)
#
#             # print("Start Encoder")
#             # print(current.shape[1])
#             # print(max_size)
#             # print(current.shape[1])
#             depth = layer_depth(current.shape.as_list()[1] / 2)
#
#             is_last_layer = not current.shape[1] > max_size * 2
#
#             if output_channels is not None and is_last_layer:
#                 depth = output_channels
#
#             activation = "selu"
#             if not output_rectified and is_last_layer:
#                 activation = None
#
#             current, before_activation = conv(current, depth, stride=2, reuse=reuse, activation=activation)
#
#             # print(current.shape)
#             layers.append(current)
#             i += 1
#     return current, layers
#
#
# # def discriminator(current, name="discriminator", reuse=False, max_size=1, dropout=False):
# #     current = encoder(current, name=name, reuse=reuse, max_size=max_size, dropout=dropout)
# #     current = current
# #     return current
#
# def decoder(current, encoder_layers, reuse=False, dropout=True, output_channels=3, output_rectified=True):
#     i = 0
#     # with tf.variable_scope(name):
#     while current.shape[1] < output_size:
#         with tf.variable_scope("decoder_" + str(i)):
#             depth = layer_depth(current.shape.as_list()[1] * 2)
#
#             is_last_layer = current.shape[1] == output_size / 2
#
#             if is_last_layer:
#                 depth = output_channels
#
#             if current.shape[1] <= 64:
#                 # tf.cond(tf.equal(state, "train"), lambda: True, lambda: False)
#                 current = tf.cond(tf.equal(dropout, True), lambda: tf.nn.dropout(current, 0.4), lambda: current)
#
#             print("input size " + str(current.shape))
#
#             activation = "selu"
#             if not output_rectified and is_last_layer:
#                 activation = None
#
#             # current = conv(current, depth, stride=2, reuse=reuse, activation=activation)
#
#             current = deconv(current, depth, reuse=reuse, activation=activation)
#             print("output size " + str(current.shape))
#             i += 1
#             # and current.shape[1] <= 8
#             if i < len(encoder_layers):
#                 current = tf.concat((current, encoder_layers[len(encoder_layers) - 1 - i]), axis=3)
#                 # print("output skip " + str(current.shape))
#     return current
#
#
# def saveImage(image, name, resize_to=output_size, rescale_space="negative_one_to_one"):
#     # print(str(image.shape))
#     # if rescale_space == "negative_one_to_one":
#
#     image = image * 255.0
#     image = image.astype('uint8')
#
#     if image.shape[2] is 1:
#         image = np.broadcast_to(image, (image.shape[0], image.shape[1], 3))
#
#     # print("save_image " + str(image.shape))
#
#     image = Image.fromarray(image)
#
#     if resize_to != -1:
#         image = image.resize((resize_to, resize_to), Image.BILINEAR)
#
#     # view_path = name
#     image.save(name)
#
#
# def load_images(path, size_x, size_y):
#     file_names = tf.train.match_filenames_once(os.path.join(path, "*.*"))
#     print(file_names)
#     # with tf.Session() as sess:
#     #     print(filenames.eval())
#     filename_queue = tf.train.string_input_producer(file_names, shuffle=False, seed=42)
#
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     images = tf.image.decode_png(image_file)
#     images.set_shape((size_x, size_y, 3))
#     images = tf.cast(images, tf.float32)
#     images = (images / 255.0)
#     # images = encode_color(images)
#
#     return images
#
#
# def export_images(images, path, rescale_space="negative_one_to_one"):
#     for v in range(min(images.shape[0], max_preview_export)):
#         image_path = path.replace("%", str(v))
#         saveImage(images[v, :, :, :], image_path, rescale_space=rescale_space)
#
#

def create_generator(current):
    print(current.shape)

    i = 0
    while current.shape[1] > 1:
        with tf.variable_scope("conv_" + str(i)):
            depth = min(8 * number_generator_filters, 4 + i * number_generator_filters / 2)
            if current.shape[1] == 2:
                depth = 1
            current = conv(current, depth)
            print(current.shape)
        i += 1

    return current

def train():
    raw_input = tf.placeholder(tf.float32, [process_window + (batch_size - 1) + 1, 1])

    inputs = []
    targets = []

    for i in range(batch_size):
        input_local = raw_input[i: i + process_window, :]
        target_local = raw_input[i + process_window: i + process_window + 1, :]

        input_local = tf.expand_dims(input_local, axis=0)
        target_local = tf.expand_dims(target_local, axis=0)

        inputs.append(input_local)
        targets.append(target_local)

    input = tf.concat(inputs, axis=0)
    target = tf.concat(targets, axis=0)
    print("input")
    print(input)
    print(target)
#
#     # input = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
#     # output = tf.placeholder(tf.float32, [None, output_size, output_size, 3])
#
    current = create_generator(input)
    # current = tf.tanh(current)

    generated_output = current

    loss = tf.reduce_mean(tf.abs(generated_output - target))
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)

        while global_step.eval() < max_epochs:
            for dir_name, _, file_list in os.walk(data_input):
                for file_name in file_list:
                    path = dir_name + "\\" + file_name
                    path = path.replace('\\', '/')

                    step = global_step.eval()

                    print("step " + str(step))

                    sample_rate, audio = wav.read(path)
                    audio = audio[:, 0]
                    audio = audio.astype(np.float32)
                    audio /= 32768
                    # audio = audio * 2.0 - 1.0
                    audio *= 20
                    # print("max " + str(np.max(audio)))
                    # print("min " + str(np.min(audio)))

                    # Left Channel
                    # print("audio.shape " + str(audio.shape))
                    # print("audio.shape " + str(audio.shape))
                    # audio = np.expand_dims(audio, axis=0)
                    audio = np.expand_dims(audio, axis=1)

                    # offset = 0

                    # print(audio.shape)
                    # print(audio)

                    # print("shape " + str(offset < audio.shape[1] - process_window - batch_size))

                    for offset in range(0, audio.shape[0] - process_window - batch_size - 1, batch_size):
                        input_audio = audio[offset:offset + process_window + batch_size, :]
                        # print("offset " + str(offset))
                        # print(input_audio.shape)

                        # train_step.run(feed_dict={raw_input: input_audio})
                        values = sess.run({
                            "train_step": train_step,
                            "global_step": global_step,
                            "loss": loss,
                            "input": input,
                            "target": target,
                            "generated_output": generated_output
                        }, feed_dict={raw_input: input_audio})

                        print("loss " + str(values["loss"]))
                        # print("input " + str(values["input"]))
                        print("target " + str(values["target"][0, 0, 0]))
                        print("generated_output " + str(values["generated_output"][0, 0, 0]))
                        # print(values["cost"])

                    if i % 500 == 0:
                        Helper.validate_directory(save_dir)
                        _ = saver.save(sess, save_dir + "/model.ckpt", global_step=global_step)

        #     raw_input


        # train_value, global_step_value = sess.run({
        #     "train": train_group,
        #     "global_step": global_step
        # }, feed_dict={state: "train"})

#         state = tf.placeholder(tf.string)
#
        # with tf.variable_scope("input"):

#             train_images = load_images(data_train, output_size, input_size + output_size)
#             test_images = load_images(data_test, output_size, input_size + output_size)
#             # output_images, output_filenames = load_images(data_train + "/output", output_size)
#
#             # print(input_images.shape)
#
#             # input_images = tf.cond(isValidation, test_images)
#
#             train_input = tf.train.batch([train_images], batch_size=batch_size)
#             test_input = tf.train.batch([test_images], batch_size=max_preview_export)
#
#             combined_input = tf.cond(tf.equal(state, "train"), lambda: train_input, lambda: test_input)
#             # tf.placeholder(tf.float32, train_input.shape)
#
#             # combined_input = tf.cond()
#             print("combined_input shape " + str(combined_input.shape))
#
#             input = combined_input[:, :input_size, :input_size, :]
#             output = combined_input[:, :output_size, input_size:, :]
#
#             print("input shape " + str(input.shape))
#             print("output shape " + str(output.shape))
#
#             input = encode_color(input, encoding=input_color_space)
#             output = encode_color(output, encoding=output_color_space)
#
#         # print(combined_input.shape)
#         # print(input.shape)
#         # print(output.shape)
#
#         current = input
#
#         # current = tf.random_normal([64, 64], mean=0, stddev=1)
#
#         # print("input shape " + current.shape)
#         # print("current shape " + current.shape)
#
#         with tf.variable_scope("generator"):
#             (encoder_current, encoder_layers) = encoder(current)
#             current = encoder_current
#
#             # current = tf.nn.dropout(current, 0.4)
#
#             print("------- middle hidden layer shape ---------")
#             print(current.shape)
#
#             dropout_enabled = tf.equal(state, "train")
#             current = decoder(current, encoder_layers, output_rectified=False, dropout=dropout_enabled,
#                               output_channels=output.shape[3])
#
#             current = tf.tanh(current)
#
#             generated_output = current
#
#         real_input = tf.concat([current, output], axis=3)
#         fake_input = tf.concat([current, generated_output], axis=3)
#
#         print("discriminator input shape " + str(real_input.shape))
#
#         with tf.variable_scope("discriminator"):
#             predict_real, _ = encoder(real_input, reuse=False, max_size=GAN_output_size, dropout=True,
#                                       output_channels=1, output_rectified=False)
#             predict_fake, _ = encoder(fake_input, reuse=True, max_size=GAN_output_size, dropout=True,
#                                       output_channels=1, output_rectified=False)
#
#             predict_real = tf.sigmoid(predict_real)
#             predict_fake = tf.sigmoid(predict_fake)
#
#         # print("Create discriminator finished")
#         # print("discriminator shape %s" % str(predict_real.shape))
#         # cost = tf.reduce_mean(abs(generated_latent - output_latent) * abs(generated_latent - output_latent))
#
#         print(tf.abs(generated_output - output).shape)
#         # l1_loss = tf.reduce_mean(tf.abs(generated_output - output))
#         # print(str(l1_loss.shape))
#         # gan_loss = tf.reduce_mean(-tf.log(predict_real) + tf.log(1 - predict_fake))
#
#         with tf.name_scope("discriminator_loss"):
#             discriminator_loss = tf.reduce_mean(-(tf.log(predict_real + epsilon) + tf.log(1 - predict_fake + epsilon)))
#             discriminator_loss = discriminator_loss * learning_rate_GAN
#
#         with tf.name_scope("generator_loss"):
#             gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + epsilon))
#             gen_loss_L1 = tf.reduce_mean(tf.abs(generated_output - output))
#
#             if output_color_space is "one_hot_h":
#                 # error = generated_output - output
#                 sv = output[:, :, :, :2] - generated_output[:, :, :, :2]
#                 h_labels = output[:, :, :, 2: 2 + color_one_hot_dim]
#                 h_logits = generated_output[:, :, :, 2: 2 + color_one_hot_dim]
#
#                 print("h_labels " + str(h_labels.shape))
#                 print("h_logits " + str(h_logits.shape))
#
#                 h_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=h_labels, logits=h_logits)
#                 h_loss = tf.reduce_mean(h_loss)
#                 sv_loss = tf.reduce_mean(tf.abs(sv))
#                 gen_loss_L1 = sv_loss + h_loss
#                 # sv = hsv[:, :, :,  1:3]
#
#             # generator_loss = gen_loss_GAN * 1 + gen_loss_L1 * 100
#             # generator_loss = gen_loss_L1
#             # gen_loss_L1 = tf.reduce_mean(tf.abs(generated_output - output))
#             generator_loss = gen_loss_GAN * learning_rate_GAN + gen_loss_L1 * learning_rate_L1
#
#         with tf.name_scope("discriminator_train"):
#             disc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
#             # print(len(disc_tvars))
#             disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
#             disc_grads_and_vars = disc_optimizer.compute_gradients(discriminator_loss, var_list=disc_tvars)
#             discriminator_train = disc_optimizer.apply_gradients(disc_grads_and_vars)
#
#         with tf.name_scope("generator_train"):
#             with tf.control_dependencies([discriminator_train]):
#                 tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#                 gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
#                 gen_grads_and_vars = gen_optimizer.compute_gradients(generator_loss, var_list=tvars)
#                 generator_train = gen_optimizer.apply_gradients(gen_grads_and_vars)
#
#         global_step = tf.contrib.framework.get_or_create_global_step()
#         incr_global_step = tf.assign(global_step, global_step + 1)
#
#         train_group = tf.group(generator_train, incr_global_step)
#         # train_group = tf.group(incr_global_step, generator_train, discriminator_train)
#
#         # cost = l1_loss + gan_loss * tf.minimum(0.1, 0.01 + tf.cast(global_step, tf.float32) / 500 * 0.1)
#         # cost = l1_loss + gan_loss * 0.01
#
#         # cost += tf.reduce_mean(abs(1 - generated_latent)) * 0.2
#         # cost += tf.reduce_mean(abs(output_latent)) * 0.2
#
#         # train_step = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)
#         # train_step = tf.train.AdamOptimizer(0.0002).minimize(generator_loss, global_step=global_step)
#
#         input = decode_color(input, encoding=input_color_space)
#         output = decode_color(output, encoding=output_color_space)
#         generated_output = decode_color(generated_output, encoding=output_color_space)
#
#         current_test = 0
#
#         # Summary
#         with tf.name_scope("summary"):
#             tf.summary.image("input", input)
#             tf.summary.image("generated_output", generated_output)
#             tf.summary.image("output", output)
#             tf.summary.scalar("gen_loss_L1", gen_loss_L1)
#             tf.summary.scalar("gen_loss_GAN", gen_loss_GAN)
#             tf.summary.scalar("discriminator_loss", discriminator_loss)
#
#         summary_merged = tf.summary.merge_all()
#
#         writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
#         tf.global_variables_initializer().run()
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
#
#         latest_checkpoint = tf.train.latest_checkpoint(save_dir)
#         if latest_checkpoint is not None and not train_from_scratch:
#             print("Loading checkpoint " + latest_checkpoint)
#             saver.restore(sess, latest_checkpoint)
#
#         while global_step.eval() < max_epochs:
#
#             i = global_step.eval()
#             # print(global_step)
#             # print()
#             # images = tf.train.shuffle_batch(
#             #     [image],
#             #     batch_size=batch_size,
#             #     num_threads=num_preprocess_threads,
#             #     capacity=min_queue_examples + 3 * batch_size,
#             #     min_after_dequeue=min_queue_examples)
#
#             # print(input_batch.shape)
#
#             # output_batch = tf.train.batch([], batch_size=batch_size)
#             # image_input = get_batch(data_train + "/input", offset=i * batch_size, size=batch_size)
#             # image_output = get_batch(data_train + "/output", offset=i * batch_size, size=batch_size)
#
#             # input.sha
#             # feed_dict = {input: input_batch, output: output_batch}
#             # output_values = sess.run(generated_output)
#             # train_step.
#             # train_step.run()
#             # sess.run()
#             # train_group.run()
#
#             train_value, global_step_value = sess.run({
#                 "train": train_group,
#                 "global_step": global_step
#             }, feed_dict={state: "train"})
#             # train_step.run()
#
#             # print(sess.run(global_step))
#             # sess.r
#             # train_step.ev
#
#             if i % 10 == 0:
#                 print(i)
#
#             test_skip = 50
#             if i > 3000:
#                 test_skip = 200
#             if i > 8000:
#                 test_skip = 500
#
#             # test_skip = 10
#
#             # print("----------------------------")
#             # print("input names ---")
#             # print("output names ---")
#             # print(input_filenames.eval())
#             # print(output_filenames.eval())
#
#             if i % test_skip == 0:
#                 print("export view")
#                 # batch = 16
#                 # test_input = get_batch(data_test + "/input", offset=current_test * batch, size=batch)
#                 # test_output = get_batch(data_test + "/output", offset=current_test * batch, size=batch)
#
#                 # input_values = input.eval()
#                 # output_values = output.eval()
#                 # generated_values = generated_output.eval()
#
#                 input_values, output_values, generated_values, real_values, fake_values, summary = sess.run(
#                     [input, output, generated_output, predict_real, predict_fake, summary_merged], feed_dict={state: "test"})
#
#                 writer.add_summary(summary, i)
#                 # generated_values2 = input.eval()
#                 # generated_values3 = input.eval()
#
#                 validate_directory(preview_dir)
#
#                 export_images(input_values, preview_dir + "/" + str(i) + " image % 0 input.png")
#                 export_images(output_values, preview_dir + "/" + str(i) + " image % 1 output.png")
#                 export_images(generated_values, preview_dir + "/" + str(i) + " image % 2 generated.png")
#                 export_images(real_values, preview_dir + "/" + str(i) + " image % 2 real.png", rescale_space=None)
#                 export_images(fake_values, preview_dir + "/" + str(i) + " image % 3 fake.png", rescale_space=None)
#
#                 # export_images(generated_values2, preview_dir + "/" + str(i) + " image % 2 generated2.png")
#                 # export_images(generated_values3, preview_dir + "/" + str(i) + " image % 2 generated3.png")
#
#                 # for v in range(exports.shape[0]):
#                 #     validate_directory(preview_dir)
#                 #     basePath = preview_dir + "/batch " + str(i) + " image " + str(v)
#                 #     # saveImage(test_input[v, :, :, :], basePath + " 0 input.png", resize_to=output_size)
#                 #     saveImage(exports[v, :, :, :], basePath + " 1 generate.png")
#                 #     # saveImage(test_output[v, :, :, :], basePath + " 2 truth.png")
#
#             current_test += 1
#
#             if i > 300 and i % 100 == 0:
#                 validate_directory(save_dir)
#                 save_path = saver.save(sess, save_dir + "/model.ckpt", global_step=global_step)
#                 print("Model saved to " + save_path)
#
#     coord.request_stop()
#     coord.join(threads)
#     sess.close()
#     # print(export)
#
#     # print(export.shape)
#
#     # prediction = tf.argmax(logits, 1)
#     # best = sess.run([prediction], feed_dict)
#     # print(best)


# generate_data_shapes()
train()
