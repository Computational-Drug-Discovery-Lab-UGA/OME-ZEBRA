import tensorflow as tf
import numpy as np
import pandas as pd
import skimage
from skimage.io import imread, imsave

np.random.seed(0)

A_orig = np.empty([500, 512 * 512], dtype=np.float32)

for i in range(0, 500):

    print(i)

    im = imread('images/DS_' + str(i) + '.tif')

    newIm = im

    # newIm = newIm.astype(np.int)
    #
    # newIm = skimage.img_as_float(im)

    newIm = newIm.flatten()

    A_orig[i, ] = newIm

A_orig = A_orig.astype(np.float32)

print("Loaded images")

A_orig_df = pd.DataFrame(A_orig)

print("Created array")

# A_df_masked = A_orig_df.copy()
# A_df_masked.iloc[0,0]=np.NAN
#
# print("Mask made")
#
# np_mask = A_df_masked.notnull()
#
# # Boolean mask for computing cost only on valid (not missing) entries
# tf_mask = tf.Variable(np_mask.values)
#
A = tf.constant(A_orig_df.values)
shape = A_orig_df.values.shape
#
# print("TF mask made")

#latent factors
rank = 100

# Initializing random H and W
temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
temp_H = np.divide(temp_H, temp_H.max())

print("H matrix made")

temp_W = np.random.randn(shape[0], rank).astype(np.float32)
temp_W = np.divide(temp_W, temp_W.max())

print("W matrix made")

H =  tf.Variable(temp_H)
W = tf.Variable(temp_W)
WH = tf.matmul(W, H)

#cost of Frobenius norm
cost = tf.reduce_mean(tf.pow(A - WH, 2))

# Learning rate
lr = 0.1
train_step = tf.train.AdamOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()

# Clipping operation. This ensure that W and H learnt are non-negative
clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
clip = tf.group(clip_W, clip_H)

print("Starting")

steps = 1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip)
        if i%10==0:
            print("\nCost: %f" % sess.run(cost))
            print("*"*40)
    learnt_W = sess.run(W)
    learnt_H = sess.run(H)

learnt_H

learnt_W

pred = np.dot(learnt_W, learnt_H)
pred_df = pd.DataFrame(pred)
pred_df.round()

A_orig_df
