'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from tensorflow.keras.applications import ResNet50, VGG16, VGG19
from tensorflow.keras.layers import Input
import imageio
import tensorflow as tf

from configs import bcolors
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in ImageNet dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', nargs='+', type=int, default=(0, 0),
                    help="occlusion upper left corner coordinate")
parser.add_argument('-occl_size', '--occlusion_size', nargs='+', type=int, default=(10, 10),
                    help="occlusion size")


args = parser.parse_args()
args.start_point = tuple(args.start_point)
args.occlusion_size = tuple(args.occlusion_size)
# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)
model2 = VGG19(input_tensor=input_tensor)
model3 = ResNet50(input_tensor=input_tensor)
# init coverage table
cov1, cov2, cov3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs
img_paths = image.list_pictures('./ImageNet/seeds/', ext='jpeg')
for _ in range(args.seeds):
    orig_np = preprocess_image(random.choice(img_paths))
    gen_img = tf.Variable(orig_np, dtype=tf.float32)

    # initial predictions
    p1 = model1.predict(gen_img.numpy())
    p2 = model2.predict(gen_img.numpy())
    p3 = model3.predict(gen_img.numpy())
    l1, l2, l3 = map(lambda p: np.argmax(p[0]), [p1, p2, p3])

    # check for early divergence
    if not (l1 == l2 == l3):
        # print(bcolors.OKGREEN +
        #         f'Already diff: {decode_label(p1)} / {decode_label(p2)} / {decode_label(p3)}' +
        #         bcolors.ENDC)
        update_coverage(gen_img.numpy(), model1, cov1, args.threshold)
        update_coverage(gen_img.numpy(), model2, cov2, args.threshold)
        update_coverage(gen_img.numpy(), model3, cov3, args.threshold)
        continue

    orig_label = l1
    name1, idx1 = neuron_to_cover(cov1)
    name2, idx2 = neuron_to_cover(cov2)
    name3, idx3 = neuron_to_cover(cov3)

    mid1 = tf.keras.Model(inputs=model1.input,
                            outputs=model1.get_layer(name1).output)
    mid2 = tf.keras.Model(inputs=model2.input,
                            outputs=model2.get_layer(name2).output)
    mid3 = tf.keras.Model(inputs=model3.input,
                            outputs=model3.get_layer(name3).output)

    for _ in range(args.grad_iterations):
        with tf.GradientTape() as tape:
            tape.watch(gen_img)
            out1 = model1(gen_img, training=False)
            out2 = model2(gen_img, training=False)
            out3 = model3(gen_img, training=False)
            # differential loss
            if args.target_model == 0:
                loss1 = -args.weight_diff * tf.reduce_mean(out1[..., orig_label])
                loss2 = tf.reduce_mean(out2[..., orig_label])
                loss3 = tf.reduce_mean(out3[..., orig_label])
            elif args.target_model == 1:
                loss1 = tf.reduce_mean(out1[..., orig_label])
                loss2 = -args.weight_diff * tf.reduce_mean(out2[..., orig_label])
                loss3 = tf.reduce_mean(out3[..., orig_label])
            else:
                loss1 = tf.reduce_mean(out1[..., orig_label])
                loss2 = tf.reduce_mean(out2[..., orig_label])
                loss3 = -args.weight_diff * tf.reduce_mean(out3[..., orig_label])
            # neuron coverage loss
            ln1 = mid1(gen_img)[..., idx1]
            ln2 = mid2(gen_img)[..., idx2]
            ln3 = mid3(gen_img)[..., idx3]
            loss_nc = args.weight_nc * (
                tf.reduce_mean(ln1) + tf.reduce_mean(ln2) + tf.reduce_mean(ln3)
            )
            total_loss = loss1 + loss2 + loss3 + loss_nc
        grads = tape.gradient(total_loss, gen_img)
        grads = normalize(grads)
        g_np = grads.numpy()
        if args.transformation == 'light':
            g_np = constraint_light(g_np)
        elif args.transformation == 'occl':
            g_np = constraint_occl(g_np, tuple(args.start_point), tuple(args.occlusion_size))
        else:
            g_np = constraint_black(g_np)
        gen_img.assign_add(tf.convert_to_tensor(g_np) * args.step)

        p1 = model1.predict(gen_img.numpy())
        p2 = model2.predict(gen_img.numpy())
        p3 = model3.predict(gen_img.numpy())
        l1, l2, l3 = map(lambda p: np.argmax(p[0]), [p1, p2, p3])
        if not (l1 == l2 == l3):
            update_coverage(gen_img.numpy(), model1, cov1, args.threshold)
            update_coverage(gen_img.numpy(), model2, cov2, args.threshold)
            update_coverage(gen_img.numpy(), model3, cov3, args.threshold)
            break