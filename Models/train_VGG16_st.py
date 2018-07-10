################### LOAD PACKAGES ###################

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import random
import argparse
import regex as re
from skimage.transform import resize

from spatial_transformer import transformer
from tf_utils import  weight_variable, bias_variable, dense_to_one_hot

data_generator = tf.keras.preprocessing.image.ImageDataGenerator
pool = tf.keras.layers.MaxPooling2D
conv = tf.keras.layers.Convolution2D
dense = tf.keras.layers.Dense
relu = tf.keras.activations.relu
flatten = tf.keras.layers.Flatten
dropout = tf.keras.layers.Dropout
glob_avg_pool = tf.keras.layers.GlobalAvgPool2D
preprocess_input = tf.keras.applications.vgg16.preprocess_input
VGG16 = tf.keras.applications.vgg16.VGG16
regularizer = tf.keras.regularizers.l2(l = 0.0005) # scale inspired by VGG16-paper


################### HELPER FUNCTIONS ###################

def get_last_epoch(model):
    file = open(model, "r")
    text = file.read()

    pattern = "Epoch [0-9]+"
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
parser.add_argument('-ba', '--bal', nargs='?', type=bool, default=False, help='Use balanced training dataset')

parser.add_argument('-t', '--train', nargs='?', type=int, default=13232, help='Training size')
parser.add_argument('-tb', '--train_bal', nargs='?', type=int, default=15664, help='Training size')
parser.add_argument('-v', '--valid', nargs='?', type=int, default=1654, help='Validation size')
parser.add_argument('-te', '--test', nargs='?', type=int, default=1657, help='Test size')

parser.add_argument('-m', '--modp', nargs='?', type=str, default='../Models/', help='Path for models')
parser.add_argument('-rm', '--resmod', nargs='?', type=str, default=' ', help='Name of restored model')
parser.add_argument('-l', '--logp', nargs='?', type=str, default='../Logs/', help='Path for logs')
parser.add_argument('-s', '--size', nargs='?', type=int, default=250, help='Size of images')
parser.add_argument('-o', '--out_size', nargs='?', type=int, default=224, help='Out size of images')
parser.add_argument('-d', '--dscl', nargs='?', type=int, default=150, help='Downscale image for STN')
parser.add_argument('-c', '--chns', nargs='?', type=int, default=3, help='Number of color channels')
parser.add_argument('-n', '--ncls', nargs='?', type=int, default=5, help='Number of classes')
parser.add_argument('-ta', '--tasks', nargs='?', type=int, default=3, help='Number of tasks')
parser.add_argument('-nn', '--net_name', nargs='?', type=str, default="vgg16_newst", help='Name of the network')
parser.add_argument('-e', '--epochs', nargs='?', type=int, default=200, help='Number of epochs')
parser.add_argument('-b', '--batch_size', nargs='?', type=int, default=64, help='Batch size')
parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.0001, help='Learning rate')

############### TRAIN DATA ###############
## Psoriasis: 0-5235
## Eczema: 5236-9518
## Acne_vulgaris: 9519-9982
## Rosacea: 9983-11266
## Mycosis_fungoides: 11267-13234

############### VALID DATA ###############
## Psoriasis: 0-653
## Eczema: 654-1188
## Acne_vulgaris: 1189-1246
## Rosacea: 1247-1407
## Mycosis_fungoides: 1408-1653

############### TEST DATA ###############
## Psoriasis: 0-654
## Eczema: 655-1190
## Acne_vulgaris: 1191-1249
## Rosacea: 1250-1410
## Mycosis_fungoides: 1411-1657

args = parser.parse_args()
path = args.path
balanced = args.bal


mod_path = args.modp
resmod = args.resmod
log_path = args.logp
img_s = args.size
out_size = args.out_size
down_scale = args.dscl
n_channels = args.chns
n_classes = args.ncls
n_tasks = args.tasks
net_name = args.net_name
n_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

date_time = str(datetime.now())
date = date_time[5:10]
time = date_time[11:16]

if balanced:
    train_size = args.train_bal
else:
    train_size = args.train
val_size = args.valid
te = args.test

if not os.path.exists(mod_path):
    os.makedirs(mod_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

path_to_mod = '{}{}-{}-{}.ckpt'.format(mod_path, date, time, net_name)
path_to_res_mod = '{}{}.ckpt'.format(mod_path,resmod)
path_to_log = '{}{}-{}-{}.txt'.format(log_path, date, time, net_name)
path_to_log_t0 = '{}{}-{}-{}_t0.txt'.format(log_path, date, time, net_name)
path_to_log_t1 = '{}{}-{}-{}_t1.txt'.format(log_path, date, time, net_name)
path_to_log_t2 = '{}{}-{}-{}_t2.txt'.format(log_path, date, time, net_name)
path_to_log_stn = '{}{}-{}-{}_stn.txt'.format(log_path, date, time, net_name)
path_to_res_log = '{}{}.txt'.format(log_path,resmod)

################### BEGIN LOGGING ###################

f = open(path_to_log, 'w')
f_t0 = open(path_to_log_t0, 'w')
f_t1 = open(path_to_log_t1, 'w')
f_t2 = open(path_to_log_t2, 'w')
f_stn = open(path_to_log_stn, 'w')

print("Path to input images:", path, file = f)
print("Size of input images:", img_s, file = f)
print("Number of color channels:", n_channels, file = f)
print("Number of classes:", n_classes, file = f)
print("Path to input images:", path)
print("Size of input images:", img_s)
print("Number of color channels:", n_channels)
print("Number of classes:", n_classes)

################### LOAD DATA ###################

if balanced:
    X_train = np.memmap('{}/X_train_bal'.format(path), dtype='float32', mode='c', shape=(train_size, img_s, img_s, n_channels))
    print("Using balanced training data", file = f)
    print("Using balanced training data")
else:
    X_train = np.memmap('{}/X_train'.format(path), dtype='float32', mode='c', shape=(train_size, img_s, img_s, n_channels))
print('X_train done')

X_valid = np.memmap('{}/X_valid'.format(path), dtype='float32', mode='c', shape=(val_size, img_s,img_s, n_channels))
print('X_valid done')

if balanced:
    Y_train = np.memmap('{}/Y_train_bal'.format(path), dtype='int', mode='c', shape=(train_size, 1))
else:
    Y_train = np.memmap('{}/Y_train'.format(path), dtype='int', mode='c', shape=(train_size, 1))
print('Y_train done')

Y_valid = np.memmap('{}/Y_valid'.format(path), dtype='int', mode='c', shape=(val_size, 1))
print('Y_valid done')

PT_model = VGG16(weights = "imagenet",
                 include_top = False,
                 input_tensor = None,
                 input_shape = (img_s, img_s, n_channels),
                 pooling = None)

PT_model.layers.pop()

for i, layer in enumerate(PT_model.layers):
    if i < 15:
        layer.trainable = False

x = tf.placeholder(tf.float32, [None, img_s, img_s, n_channels])
x_low = tf.placeholder(tf.float32, [None, down_scale, down_scale, n_channels])
m = tf.placeholder(tf.float32, [None, n_tasks])
y = tf.placeholder(tf.float32, [None, n_tasks])
dropout_rate = tf.placeholder(tf.float32)

step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(learning_rate, step, 5, 0.9999)

######################################### VGG16 NETWORK IMAGENET WEIGHTS AND SPATIAL TRANSFORMER #########################################
with tf.device('/gpu:0'):
    with tf.variable_scope('network'): 
        ######################################### LOCALISATION NETWORK #########################################

        x_loc_c1 = conv(32, (3, 3), activation='relu', padding='same', name='loc_conv1')(x_low)
        x_loc_p1 = pool((5, 5), strides=(5, 5), name='loc_pool1')(x_loc_c1)
        x_loc_c2 = conv(32, (3, 3), activation='relu', padding='same', name='loc_conv2')(x_loc_p1)
        x_loc_p2 = pool((5, 5), strides=(5, 5), name='loc_pool2')(x_loc_c2)
        
        x_loc_f1 = flatten(name='loc_flatten1')(x_loc_p2)
        x_loc_de1 = dense(64, activation='relu', name='loc_fc1')(x_loc_f1)

        W_fc_loc2 = weight_variable([64, 6])
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
        h_fc_loc2 = tf.nn.tanh(tf.matmul(x_loc_de1, W_fc_loc2) + b_fc_loc2)

        ######################################### SPATIAL TRANSFORMER LAYER #########################################
        o_size = (out_size, out_size)
        h_trans = transformer(x, h_fc_loc2, o_size)
        h_trans.set_shape([None, out_size, out_size, n_channels])

        ######################################### PRE-TRAINED NETWORK #########################################
        PT_layer = PT_model(h_trans)

        x_gap = glob_avg_pool(name='glob_avg_pool')(PT_layer)

        # # %% flatten
        x_f1 = flatten(name='flatten')(x_gap)
        x_de = dense(64, activation='relu', name='fc1', kernel_regularizer = regularizer)(x_f1)
        x_d6 = dropout(rate = dropout_rate, name='drop6')(x_de)

        # %% IN CASE OF PSORIASIS OR ECZEMA
        x_de0 = dense(32, activation='relu', name='fc01', kernel_regularizer = regularizer)(x_d6)
        y_logits0 = dense(1, name='predictions0')(x_de0)

        # %% IN CASE OF ACNE VULGARIS OR ROSACEA
        x_de1 = dense(32, activation='relu', name='fc11', kernel_regularizer = regularizer)(x_d6)
        y_logits1 = dense(1, name='predictions1')(x_de1)

        # %% IN CASE OF ECZEMA OR MYCOSIS FUNGOIDES
        x_de2 = dense(32, activation='relu', name='fc21', kernel_regularizer = regularizer)(x_d6)
        y_logits2 = dense(1, name='predictions2')(x_de2)

        y_logits = tf.concat([y_logits0, y_logits1, y_logits2], 1)

        n_samples = tf.reduce_sum(tf.count_nonzero(m, axis=1, dtype=tf.float32))

    with tf.variable_scope('loss'):
        masked_cross_entropy = masked_sigmoid_cross_entropy_with_logits(logits = y_logits, labels = y, masks = m)
        l2_loss = tf.losses.get_regularization_loss()
        loss_unreg = tf.reduce_sum(masked_cross_entropy)/n_samples
        loss = loss_unreg + l2_loss

    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables), global_step = step)

    with tf.variable_scope('performance'):
        prediction = tf.round(tf.multiply(tf.sigmoid(y_logits), m))
        correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(prediction, y), dtype=tf.float32), m))
        accuracy = correct_prediction/n_samples

    with tf.variable_scope('tasks'):
        t0 = tf.constant([1, 0, 0], dtype=tf.float32)   # psoriasis
        t1 = tf.constant([0, 1, 0], dtype=tf.float32)   # acne ans rosacea
        t2 = tf.constant([0, 0, 1], dtype=tf.float32)   # mycosis
        t02 = tf.constant([1, 0, 1], dtype=tf.float32)  # eczema
       
        b0 = tf.reduce_all(tf.equal(m, t0), axis=1)     # psoriasis
        b1 = tf.reduce_all(tf.equal(m, t1), axis=1)     # acne and rosacea
        b2 = tf.reduce_all(tf.equal(m, t2), axis=1)     # mycosis
        b02 = tf.reduce_all(tf.equal(m, t02), axis=1)   # eczema

        i0 = tf.concat([tf.where(b0), tf.where(b02)],axis=0)  # psoriasis and eczema
        i1 = tf.where(b1)                                     # acne and rosacea
        i2 = tf.concat([tf.where(b2), tf.where(b02)],axis=0)  # mycosis and eczema

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

######################################### TRAINING #########################################

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95, visible_device_list= '0'), device_count={'GPU': 1})) # , log_device_placement=True
sess.run(tf.global_variables_initializer())

## RESTORE MODEL IF IT EXISTS
if len(resmod) > 1 and os.path.exists(path_to_res_mod + ".index"):
    saver.restore(sess, path_to_res_mod)
    print("Model restored: " + resmod, file = f)
    print("Model restored: " + resmod)
    last_epoch = get_last_epoch(path_to_res_log)
    epoch_range = range(last_epoch+1,last_epoch+n_epochs+1)
else:
    epoch_range = range(n_epochs)

# Data augmentation
train_dat = data_generator(zoom_range=0.2,
                           horizontal_flip=True,
                           vertical_flip=True,
                           rotation_range=90,
                           width_shift_range=0.1,
                           height_shift_range=0.1)


iter_per_epoch = int(np.ceil(train_size/batch_size))

all_val_indices = np.arange(val_size)
n_vals = int(np.ceil(val_size/64.0))

print("Training: Running {} epochs for {} iterations with batch size {}".format(n_epochs, iter_per_epoch, batch_size), file = f)
print("Training: Running {} epochs for {} iterations with batch size {}".format(n_epochs, iter_per_epoch, batch_size))

min_val_loss = 2
patience = 0

for epoch_i in epoch_range:
    if epoch_i > 0:
        f = open(path_to_log, 'a')
        f_t0 = open(path_to_log_t0, 'a')
        f_t1 = open(path_to_log_t1, 'a')
        f_t2 = open(path_to_log_t2, 'a')
        f_stn = open(path_to_log_stn, 'a')
    print(".................................... Epoch {} ..................................".format(epoch_i), file = f)
    print(".................................... Epoch {} ..................................".format(epoch_i), file = f_t0)
    print(".................................... Epoch {} ..................................".format(epoch_i), file = f_t1)
    print(".................................... Epoch {} ..................................".format(epoch_i), file = f_t2)
    print(".................................... Epoch {} ..................................".format(epoch_i), file = f_stn)
    print(".................................... Epoch {} ..................................".format(epoch_i))

    train_loss_reg = 0

    train_pred = []
    train_cross_entropy = []

    train_pred_t0 = []
    train_pred_t1 = []
    train_pred_t2 = []

    train_cross_entropy_t0 = []
    train_cross_entropy_t1 = []
    train_cross_entropy_t2 = []

    all_indices = random.sample(range(train_size), train_size)

    for i, batch_idxs in enumerate(batch(all_indices,batch_size)):
        batch_xs = np.array([train_dat.random_transform(x) for x in X_train[batch_idxs]])
        batch_ys = Y_train[batch_idxs]
        batch_xs_low = (batch_xs.copy()).astype(np.uint8)
        batch_xs_low = (np.array([resize(x, (down_scale, down_scale)) for x in batch_xs_low])).astype(np.float32)

        batch_xs = preprocess_input(batch_xs)
        labels, masks = generate_masks_and_labels(batch_ys)

        feed = {x: batch_xs, x_low: batch_xs_low, m: masks, y: labels, dropout_rate: 0.5}

        batch_acc = sess.run(accuracy, feed_dict = feed)
        pred_t0 = sess.run(correct_prediction_t0, feed_dict = feed)
        pred_t1 = sess.run(correct_prediction_t1, feed_dict = feed)
        pred_t2 = sess.run(correct_prediction_t2, feed_dict = feed)

        pred_t0 = pred_t0.flatten()
        pred_t1 = pred_t1.flatten()
        pred_t2 = pred_t2.flatten()

        if len(pred_t0) > 0:
            train_pred_t0 += [p for p in pred_t0]
        if len(pred_t1) > 0:
            train_pred_t1 += [p for p in pred_t1]
        if len(pred_t2) > 0:
            train_pred_t2 += [p for p in pred_t2]

        train_pred += [p for p in pred_t0] + [p for p in pred_t1] + [p for p in pred_t2]


        batch_loss = sess.run(loss, feed_dict = feed)
        train_loss_reg += batch_loss

        cross_ent_t0 = sess.run(cross_entropy_t0, feed_dict = feed)
        cross_ent_t1 = sess.run(cross_entropy_t1, feed_dict = feed)
        cross_ent_t2 = sess.run(cross_entropy_t2, feed_dict = feed)

        cross_ent_t0 = cross_ent_t0.flatten()
        cross_ent_t1 = cross_ent_t1.flatten()
        cross_ent_t2 = cross_ent_t2.flatten()

        if len(cross_ent_t0) > 0:
            train_cross_entropy_t0 += [c for c in cross_ent_t0]
        if len(cross_ent_t1) > 0:
            train_cross_entropy_t1 += [c for c in cross_ent_t1]
        if len(cross_ent_t2) > 0:
            train_cross_entropy_t2 += [c for c in cross_ent_t2]

        train_cross_entropy += [c for c in cross_ent_t0] + [c for c in cross_ent_t1] + [c for c in cross_ent_t2]

        if i % 10 == 0:
            print('Iteration: %i\t Loss: %.4f\t Accuracy: %.4f' % (i, batch_loss, batch_acc), file = f)
            print('Iteration: %i\t Loss: %.4f\t Accuracy: %.4f' % (i, batch_loss, batch_acc))


        sess.run(optimize, feed_dict = feed)

    curr_learning_rate = sess.run(optimizer._lr, feed_dict = feed)

    ######## VALIDATION PERFORMANCE ########

    val_loss_reg = 0

    val_pred = []
    val_cross_entropy = []

    val_pred_t0 = []
    val_pred_t1 = []
    val_pred_t2 = []

    val_cross_entropy_t0 = []
    val_cross_entropy_t1 = []
    val_cross_entropy_t2 = []

    for i, batch_val_idxs in enumerate(batch(all_val_indices,64)):
        batch_val_xs = X_valid[batch_val_idxs]
        batch_val_ys = Y_valid[batch_val_idxs]
        batch_val_xs_low = (batch_val_xs.copy()).astype(np.uint8)
        batch_val_xs_low = (np.array([resize(x, (down_scale, down_scale)) for x in batch_val_xs_low])).astype(np.float32)

        batch_val_xs = preprocess_input(batch_val_xs)
        labels_val, masks_val = generate_masks_and_labels(batch_val_ys)

        feed = {x: batch_val_xs, x_low: batch_val_xs_low, m: masks_val, y: labels_val, dropout_rate: 0.0}

        batch_acc = sess.run(accuracy, feed_dict = feed)
        pred_t0 = sess.run(correct_prediction_t0, feed_dict = feed)
        pred_t1 = sess.run(correct_prediction_t1, feed_dict = feed)
        pred_t2 = sess.run(correct_prediction_t2, feed_dict = feed)

        pred_t0 = pred_t0.flatten()
        pred_t1 = pred_t1.flatten()
        pred_t2 = pred_t2.flatten()

        if len(pred_t0) > 0:
            val_pred_t0 += [p for p in pred_t0]
        if len(pred_t1) > 0:
            val_pred_t1 += [p for p in pred_t1]
        if len(pred_t2) > 0:
            val_pred_t2 += [p for p in pred_t2]

        val_pred += [p for p in pred_t0] + [p for p in pred_t1] + [p for p in pred_t2]

        batch_loss = sess.run(loss, feed_dict = feed)
        val_loss_reg += batch_loss

        cross_ent_t0 = sess.run(cross_entropy_t0, feed_dict = feed)
        cross_ent_t1 = sess.run(cross_entropy_t1, feed_dict = feed)
        cross_ent_t2 = sess.run(cross_entropy_t2, feed_dict = feed)

        cross_ent_t0 = cross_ent_t0.flatten()
        cross_ent_t1 = cross_ent_t1.flatten()
        cross_ent_t2 = cross_ent_t2.flatten()

        if len(cross_ent_t0) > 0:
            val_cross_entropy_t0 += [c for c in cross_ent_t0]
        if len(cross_ent_t1) > 0:
            val_cross_entropy_t1 += [c for c in cross_ent_t1]
        if len(cross_ent_t2) > 0:
            val_cross_entropy_t2 += [c for c in cross_ent_t2]

        val_cross_entropy += [c for c in cross_ent_t0] + [c for c in cross_ent_t1] + [c for c in cross_ent_t2]

        theta = sess.run(h_fc_loc2, feed_dict = feed)

        for stn in theta:
            print(stn, file = f_stn)



    train_acc = sum(train_pred)/len(train_pred)
    train_acc_t0 = sum(train_pred_t0)/len(train_pred_t0)
    train_acc_t1 = sum(train_pred_t1)/len(train_pred_t1)
    train_acc_t2 = sum(train_pred_t2)/len(train_pred_t2)

    print('Average training accuracy: %.4f' % (train_acc), file = f)
    print('Average training accuracy: %.4f' % (train_acc_t0), file = f_t0)
    print('Average training accuracy: %.4f' % (train_acc_t1), file = f_t1)
    print('Average training accuracy: %.4f' % (train_acc_t2), file = f_t2)
    print('Average training accuracy: %.4f' % (train_acc))

    val_acc = sum(val_pred)/len(val_pred)
    val_acc_t0 = sum(val_pred_t0)/len(val_pred_t0)
    val_acc_t1 = sum(val_pred_t1)/len(val_pred_t1)
    val_acc_t2 = sum(val_pred_t2)/len(val_pred_t2)

    print('Average validation accuracy: %.4f' % (val_acc), file = f)
    print('Average validation accuracy: %.4f' % (val_acc_t0), file = f_t0)
    print('Average validation accuracy: %.4f' % (val_acc_t1), file = f_t1)
    print('Average validation accuracy: %.4f' % (val_acc_t2), file = f_t2)
    print('Average validation accuracy: %.4f' % (val_acc))

    train_loss = sum(train_cross_entropy)/len(train_cross_entropy)
    train_loss_t0 = sum(train_cross_entropy_t0)/len(train_cross_entropy_t0)
    train_loss_t1 = sum(train_cross_entropy_t1)/len(train_cross_entropy_t1)
    train_loss_t2 = sum(train_cross_entropy_t2)/len(train_cross_entropy_t2)

    print('Average training loss: %.4f' % (train_loss), file = f)
    print('Average training loss: %.4f' % (train_loss_t0), file = f_t0)
    print('Average training loss: %.4f' % (train_loss_t1), file = f_t1)
    print('Average training loss: %.4f' % (train_loss_t2), file = f_t2)
    print('Average training loss: %.4f' % (train_loss))

    val_loss = sum(val_cross_entropy)/len(val_cross_entropy)
    val_loss_t0 = sum(val_cross_entropy_t0)/len(val_cross_entropy_t0)
    val_loss_t1 = sum(val_cross_entropy_t1)/len(val_cross_entropy_t1)
    val_loss_t2 = sum(val_cross_entropy_t2)/len(val_cross_entropy_t2)

    print('Average validation loss: %.4f' % (val_loss), file = f)
    print('Average validation loss: %.4f' % (val_loss_t0), file = f_t0)
    print('Average validation loss: %.4f' % (val_loss_t1), file = f_t1)
    print('Average validation loss: %.4f' % (val_loss_t2), file = f_t2)
    print('Average validation loss: %.4f' % (val_loss))

    print('Average training regularized loss: %.4f' % (train_loss_reg/iter_per_epoch), file = f)
    print('Average training regularized loss: %.4f' % (train_loss_reg/iter_per_epoch))

    print('Average validation regularized loss: %.4f' % (val_loss_reg/n_vals), file = f)
    print('Average validation regularized loss: %.4f' % (val_loss_reg/n_vals))

    print("\nCurrent learning rate: %.10f" % curr_learning_rate, file = f)
    print("\nCurrent learning rate: %.10f" % curr_learning_rate)

    print("................................................................................")

    ### SAVE WEIGHTS FROM MODEL
    if min_val_loss > val_loss:
        min_val_loss = val_loss
        save_path = saver.save(sess, path_to_mod)
        print("Model saved in path: {}".format(save_path))
        print("Model saved in path: {}".format(save_path), file = f)
        print("Best validation loss.")
        print("Best validation loss.", file=f)
        patience = 0
    else:
        patience += 1     

    f.close()
    f_t0.close()
    f_t1.close()
    f_t2.close()
    f_stn.close()

    if patience == 25:
        break

sess.close()
