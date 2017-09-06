import numpy as np
import scipy.signal
import tensorflow as tf
import imageio
import simulator.Simulator2 as Sim2


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        #print(from_var, to_var)
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def get_empty_loss_arrays(size):
    v_l = np.empty(size)
    p_l = np.empty(size)
    e_l = np.empty(size)
    g_n = np.empty(size)
    v_n = np.empty(size)
    return v_l, p_l, e_l, g_n, v_n


def make_gif(images, name, duration, width=15, height=15):
    with imageio.get_writer(name, mode='I', duration=duration) as writer:
        for image in images:
            rgb_image = np.zeros([height*Sim2.HEIGHT, width*Sim2.WIDTH, 3])
            #print(image)
            #image = image[0]
            #print(image.shape, image)
            for i in range(len(image)):
                pixel = Sim2.colors[int(image[i] * (len(Sim2.colors) - 1) + 0.01)]
                #print(i, pixel)
                for y in range(Sim2.HEIGHT):
                    for x in range(Sim2.WIDTH):
                        #print(i, (i % height)*Sim2.HEIGHT+y,int(int(i / height)*Sim2.WIDTH)+x)
                        rgb_image[(i % height)*Sim2.HEIGHT+y][int(int(i / height)*Sim2.WIDTH)+x] = pixel
            writer.append_data(rgb_image)

