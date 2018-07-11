################### LOAD PACKAGES ###################

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import random
import argparse
import regex as re

from keras.preprocessing.image import ImageDataGenerator
pool = tf.keras.layers.MaxPooling2D
conv = tf.keras.layers.Convolution2D
dense = tf.keras.layers.Dense
relu = tf.keras.activations.relu
flatten = tf.keras.layers.Flatten
dropout = tf.keras.layers.Dropout
glob_avg_pool = tf.keras.layers.GlobalAvgPool2D
preprocess_input = tf.keras.applications.vgg16.preprocess_input
VGG16 = tf.keras.applications.vgg16.VGG16

################### HELPER FUNCTIONS ###################

def get_last_epoch(model):
    file = open(model, "r")
    text = file.read()

    pattern = "Epoch \([0-9]+\)"
    results = re.findall(pattern, text)
    last_result = results[-1]
    last_epoch = re.findall("[0-9]+", last_result)[0]

    file.close()
    return int(last_epoch)

def generate_masks_and_labels(init_labels):
    masks = []
    labels = []

    label00 = label10 = label20 = [0, 0, 0]
    label021 = mask021 = [1, 0, 1]
    label11 = mask1 = [0, 1, 0]
    mask00 = [1, 0, 0]
    mask20 = [0, 0, 1]

    for label in init_labels:
        if label[0] == 0:           ### psoriasis
            masks.append(mask00)    # mask = [1, 0, 0]
            labels.append(label00)  # label = [0, 0, 0]
        elif label[0] == 1:         ### eczema 
            masks.append(mask021)   # mask = [1, 0, 1]
            labels.append(label021) # label = [1, 0, 1]
        elif label[0] == 2:         ### acne vulgaris
            masks.append(mask1)     # mask = [0, 1, 0]
            labels.append(label10)  # label = [0, 0, 0]
        elif label[0] == 3:         ### rosacea
            masks.append(mask1)     # mask = [0, 1, 0]
            labels.append(label11)  # label = [0, 1, 0]
        else:                       ### mycosis fungoides
            masks.append(mask20)    # mask = [0, 0, 1]
            labels.append(label20)  # label = [0, 0, 0]
    return np.array(labels), np.array(masks)

def masked_sigmoid_cross_entropy_with_logits(logits, labels, masks):
    return tf.multiply(masks, tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))  

def batch(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

################### ARGUMENTS AND PARAMETERS ###################

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', nargs='?', type=str, default='../Data_crop', help='Path to data')
parser.add_argument('-te', '--test', nargs='?', type=int, default=1657, help='Test size')

parser.add_argument('-m', '--modp', nargs='?', type=str, default='../Models/', help='Path for models')
parser.add_argument('-l', '--testp', nargs='?', type=str, default='../Tests/', help='Path for test logs')
parser.add_argument('-rm', '--resmod', nargs='?', type=str, default=' ', help='Name of restored model')
parser.add_argument('-s', '--size', nargs='?', type=int, default=250, help='Size of images')
parser.add_argument('-c', '--chns', nargs='?', type=int, default=3, help='Number of color channels')
parser.add_argument('-n', '--ncls', nargs='?', type=int, default=5, help='Number of classes')
parser.add_argument('-ta', '--tasks', nargs='?', type=int, default=3, help='Number of tasks')

############### TRAIN DATA ###############
## Psoriasis: 0-5235
## Eczema: 5236-9515
## Acne_vulgaris: 9516-9979
## Rosacea: 9980-11263
## Mycosis_fungoides: 11264-13231

############### VALID DATA ###############
## Psoriasis: 0-653
## Eczema: 654-1188
## Acne_vulgaris: 1189-1246
## Rosacea: 1247-1407
## Mycosis_fungoides: 1408-1653

############### TEST DATA ###############
## Psoriasis: 0-654
## Eczema: 655-1189
## Acne_vulgaris: 1190-1248
## Rosacea: 1249-1409
## Mycosis_fungoides: 1410-1656

args = parser.parse_args()
path = args.path

mod_path = args.modp
test_log_path = args.testp
resmod = args.resmod
img_s = args.size
n_channels = args.chns
n_classes = args.ncls
n_tasks = args.tasks

test_size = args.test


################### BEGIN LOGGING ###################

if not os.path.exists(test_log_path):
    os.makedirs(test_log_path)

path_to_test_log = '{}{}_test.txt'.format(test_log_path,resmod)
path_to_test_log_t0 = '{}{}_test_t0.txt'.format(test_log_path,resmod)
path_to_test_log_t1 = '{}{}_test_t1.txt'.format(test_log_path,resmod)
path_to_test_log_t2 = '{}{}_test_t2.txt'.format(test_log_path,resmod)
path_to_res_mod = '{}{}.ckpt'.format(mod_path,resmod)


f = open(path_to_test_log, 'w')
f_t0 = open(path_to_test_log_t0, 'w')
f_t1 = open(path_to_test_log_t1, 'w')
f_t2 = open(path_to_test_log_t2, 'w')

print("Path to input images:", path, file = f)
print("Size of input images:", img_s, file = f)
print("Number of color channels:", n_channels, file = f)
print("Number of classes:", n_classes, file = f)

################### LOAD DATA ###################

X_test = np.memmap('{}/X_test'.format(path), dtype='float32', mode='c', shape=(test_size, img_s, img_s, n_channels))
print('X_test done')

Y_test = np.memmap('{}/Y_test'.format(path), dtype='int', mode='c', shape=(test_size, 1))
print('Y_test done')

PT_model = VGG16(weights = "imagenet",
                 include_top = False,
                 input_tensor = None,
                 input_shape = (img_s, img_s, n_channels),
                 pooling = None)

for layer in PT_model.layers:
    layer.trainable = False

x = tf.placeholder(tf.float32, [None, img_s, img_s, n_channels])
m = tf.placeholder(tf.float32, [None, n_tasks])
y = tf.placeholder(tf.float32, [None, n_tasks])

######################################### VGG16 NETWORK IMAGENET WEIGHTS #########################################
with tf.device('/gpu:0'):
    with tf.variable_scope('network'):
        PT_layer = PT_model(x)

        # %% flatten
        x_f1 = flatten(name='flatten')(PT_layer)
        x_de = dense(256, activation='relu', name='fc1')(x_f1)

        # %% IN CASE OF PSORIASIS OR ECZEMA
        x_de0 = dense(128, activation='relu', name='fc01')(x_de)
        y_logits0 = dense(1, name='predictions0')(x_de0)

        # %% IN CASE OF ACNE VULGARIS OR ROSACEA
        x_de1 = dense(128, activation='relu', name='fc11')(x_de)
        y_logits1 = dense(1, name='predictions1')(x_de1)

        # %% IN CASE OF ECZEMA OR MYCOSIS FUNGOIDES
        x_de2 = dense(128, activation='relu', name='fc21')(x_de)
        y_logits2 = dense(1, name='predictions2')(x_de2)

        y_logits = tf.concat([y_logits0, y_logits1, y_logits2], 1)   

        n_samples = tf.reduce_sum(tf.count_nonzero(m, axis=1, dtype=tf.float32))

    with tf.variable_scope('loss'):
        masked_cross_entropy = masked_sigmoid_cross_entropy_with_logits(logits = y_logits, labels = y, masks = m)
        loss = tf.reduce_sum(masked_cross_entropy)/n_samples

    with tf.variable_scope('performance'):
        probabilities = tf.multiply(tf.sigmoid(y_logits), m)
        prediction = tf.round(probabilities)
        correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(prediction, y), dtype=tf.float32), m))
        accuracy = correct_prediction/n_samples

    with tf.variable_scope('tasks'):
        t0 = tf.constant([1, 0, 0], dtype=tf.float32)   # psoriasis
        t1 = tf.constant([0, 1, 0], dtype=tf.float32)   # acne or rosacea
        t2 = tf.constant([0, 0, 1], dtype=tf.float32)   # mycosis
        t02 = tf.constant([1, 0, 1], dtype=tf.float32)  # eczema
       
        b0 = tf.reduce_all(tf.equal(m, t0), axis=1)     # psoriasis
        b1 = tf.reduce_all(tf.equal(m, t1), axis=1)     # acne or rosacea
        b2 = tf.reduce_all(tf.equal(m, t2), axis=1)     # mycosis
        b02 = tf.reduce_all(tf.equal(m, t02), axis=1)   # eczema

        i0 = tf.concat([tf.where(b0), tf.where(b02)],axis=0)  # psoriasis or eczema
        i1 = tf.where(b1)                                     # acne or rosacea
        i2 = tf.concat([tf.where(b2), tf.where(b02)],axis=0)  # mycosis or eczema

        p0 = tf.gather(tf.slice(prediction, [0, 0], [-1, 1]), i0) # collect task 0 predictions
        p1 = tf.gather(tf.slice(prediction, [0, 1], [-1, 1]), i1) # collect task 1 predictions
        p2 = tf.gather(tf.slice(prediction, [0, 2], [-1, 1]), i2) # collect task 2 predictions

        c0 = tf.gather(tf.slice(y, [0, 0], [-1, 1]), i0)  # collect task 0 labels
        c1 = tf.gather(tf.slice(y, [0, 1], [-1, 1]), i1)  # collect task 1 labels
        c2 = tf.gather(tf.slice(y, [0, 2], [-1, 1]), i2)  # collect task 2 labels

        correct_prediction_t0 = tf.cast(tf.equal(p0, c0), dtype=tf.float32)
        correct_prediction_t1 = tf.cast(tf.equal(p1, c1), dtype=tf.float32)  
        correct_prediction_t2 = tf.cast(tf.equal(p2, c2), dtype=tf.float32)  

        cross_entropy_t0 = tf.gather(tf.slice(masked_cross_entropy, [0, 0], [-1, 1]), i0)
        cross_entropy_t1 = tf.gather(tf.slice(masked_cross_entropy, [0, 1], [-1, 1]), i1)
        cross_entropy_t2 = tf.gather(tf.slice(masked_cross_entropy, [0, 2], [-1, 1]), i2)

        probabilities_t0 = tf.gather(tf.slice(probabilities, [0, 0], [-1, 1]), i0)
        probabilities_t1 = tf.gather(tf.slice(probabilities, [0, 1], [-1, 1]), i1)
        probabilities_t2 = tf.gather(tf.slice(probabilities, [0, 2], [-1, 1]), i2)

######################################### TRAINING #########################################

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95, visible_device_list= '0'), device_count={'GPU': 1})) # , log_device_placement=True
sess.run(tf.global_variables_initializer())

## RESTORE MODEL IF IT EXISTS
if len(resmod) > 1 and os.path.exists(path_to_res_mod + ".index"):
    saver.restore(sess, path_to_res_mod)
    print("Model restored: " + resmod)

######## TEST PERFORMANCE ########

test_pred = []
test_labels_pred = []
test_labels_true = []
test_probabilities = []
test_cross_entropy = []

test_labels_pred_t0 = []
test_labels_pred_t1 = []
test_labels_pred_t2 = []

test_labels_true_t0 = []
test_labels_true_t1 = []
test_labels_true_t2 = []

test_pred_t0 = []
test_pred_t1 = []
test_pred_t2 = []

test_probabilities_t0 = []
test_probabilities_t1 = []
test_probabilities_t2 = []

test_cross_entropy_t0 = []
test_cross_entropy_t1 = []
test_cross_entropy_t2 = []

all_test_indices = np.arange(test_size)
n_tests = int(np.ceil(test_size/64.0))

for i, batch_test_idxs in enumerate(batch(all_test_indices,64)):
    batch_test_xs = X_test[batch_test_idxs]
    batch_test_ys = Y_test[batch_test_idxs]
    batch_test_xs = preprocess_input(batch_test_xs)

    labels_val, masks_val = generate_masks_and_labels(batch_test_ys)

    feed = {x: batch_test_xs, m: masks_val, y: labels_val}

    batch_acc = sess.run(accuracy, feed_dict = feed)
    labels_pred_t0, labels_true_t0, pred_t0, probs_t0 = sess.run([p0, c0, correct_prediction_t0, probabilities_t0], feed_dict = feed)
    labels_pred_t1, labels_true_t1, pred_t1, probs_t1 = sess.run([p1, c1, correct_prediction_t1, probabilities_t1], feed_dict = feed)
    labels_pred_t2, labels_true_t2, pred_t2, probs_t2 = sess.run([p2, c2, correct_prediction_t2, probabilities_t2], feed_dict = feed)

    labels_pred_t0 = labels_pred_t0.flatten()
    labels_pred_t1 = labels_pred_t1.flatten()
    labels_pred_t2 = labels_pred_t2.flatten()

    labels_true_t0 = labels_true_t0.flatten()
    labels_true_t1 = labels_true_t1.flatten()
    labels_true_t2 = labels_true_t2.flatten()

    probs_t0 = probs_t0.flatten()
    probs_t1 = probs_t1.flatten()
    probs_t2 = probs_t2.flatten()

    pred_t0 = pred_t0.flatten()
    pred_t1 = pred_t1.flatten()
    pred_t2 = pred_t2.flatten()

    if len(pred_t0) > 0:
        test_pred_t0 += [p for p in pred_t0]
        test_labels_pred_t0 += [lp for lp in labels_pred_t0]
        test_labels_true_t0 += [lt for lt in labels_true_t0]
        test_probabilities_t0 += [pb for pb in probs_t0]
    if len(pred_t1) > 0:
        test_pred_t1 += [p for p in pred_t1]
        test_labels_pred_t1 += [lp for lp in labels_pred_t1]
        test_labels_true_t1 += [lt for lt in labels_true_t1]
        test_probabilities_t1 += [pb for pb in probs_t1]
    if len(pred_t2) > 0:
        test_pred_t2 += [p for p in pred_t2]
        test_labels_pred_t2 += [lp for lp in labels_pred_t2]
        test_labels_true_t2 += [lt for lt in labels_true_t2]
        test_probabilities_t2 += [pb for pb in probs_t2]

    test_pred += [p for p in pred_t0] + [p for p in pred_t1] + [p for p in pred_t2]
    test_labels_pred += [lp for lp in labels_pred_t0] + [lp for lp in labels_pred_t1] + [lp for lp in labels_pred_t2]
    test_labels_true += [lt for lt in labels_true_t0] + [lt for lt in labels_true_t1] + [lt for lt in labels_true_t2]
    test_probabilities += [pb for pb in probs_t0] + [pb for pb in probs_t1] + [pb for pb in probs_t2]

    cross_ent_t0 = sess.run(cross_entropy_t0, feed_dict = feed)
    cross_ent_t1 = sess.run(cross_entropy_t1, feed_dict = feed)
    cross_ent_t2 = sess.run(cross_entropy_t2, feed_dict = feed)

    cross_ent_t0 = cross_ent_t0.flatten()
    cross_ent_t1 = cross_ent_t1.flatten()
    cross_ent_t2 = cross_ent_t2.flatten()

    if len(cross_ent_t0) > 0:
        test_cross_entropy_t0 += [c for c in cross_ent_t0]
    if len(cross_ent_t1) > 0:
        test_cross_entropy_t1 += [c for c in cross_ent_t1]
    if len(cross_ent_t2) > 0:
        test_cross_entropy_t2 += [c for c in cross_ent_t2]

    test_cross_entropy += [c for c in cross_ent_t0] + [c for c in cross_ent_t1] + [c for c in cross_ent_t2]


for i, l in enumerate(test_labels_pred):
    print("{} - {}".format(l, test_labels_true[i]), file=f)
for i, l in enumerate(test_labels_pred_t0):
    print("{} - {}".format(l, test_labels_true_t0[i]), file=f_t0)
for i, l in enumerate(test_labels_pred_t1):
    print("{} - {}".format(l, test_labels_true_t1[i]), file=f_t1)
for i, l in enumerate(test_labels_pred_t2):
    print("{} - {}".format(l, test_labels_true_t2[i]), file=f_t2)

for p in test_probabilities_t0:
    print("Prob: %.4f" % (p), file=f_t0)
for p in test_probabilities_t1:
    print("Prob: %.4f" % (p), file=f_t1)
for p in test_probabilities_t2:
    print("Prob: %.4f" % (p), file=f_t2)

test_acc = sum(test_pred)/len(test_pred)
test_acc_t0 = sum(test_pred_t0)/len(test_pred_t0)
test_acc_t1 = sum(test_pred_t1)/len(test_pred_t1)
test_acc_t2 = sum(test_pred_t2)/len(test_pred_t2)

print('\nTest accuracy: %.4f' % (test_acc), file=f)
print('\nTest accuracy: %.4f' % (test_acc))
print('\nTest accuracy: %.4f' % (test_acc_t0), file=f_t0)
print('\nTest accuracy: %.4f' % (test_acc_t1), file=f_t1)
print('\nTest accuracy: %.4f' % (test_acc_t2), file=f_t2)

test_loss = sum(test_cross_entropy)/len(test_cross_entropy)
test_loss_t0 = sum(test_cross_entropy_t0)/len(test_cross_entropy_t0)
test_loss_t1 = sum(test_cross_entropy_t1)/len(test_cross_entropy_t1)
test_loss_t2 = sum(test_cross_entropy_t2)/len(test_cross_entropy_t2)

print('\nTest loss: %.4f' % (test_loss), file=f)
print('\nTest loss: %.4f' % (test_loss))
print('\nTest loss: %.4f' % (test_loss_t0), file=f_t0)
print('\nTest loss: %.4f' % (test_loss_t1), file=f_t1)
print('\nTest loss: %.4f' % (test_loss_t2), file=f_t2)
sess.close()
f.close()
f_t0.close()
f_t1.close()
f_t2.close()
