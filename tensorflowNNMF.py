import tensorflow as tf
import numpy as np
import pandas as pd
np.random.seed(0)

A_orig = np.array([[3, 4, 5, 2],
                   [4, 4, 3, 3],
                   [5, 5, 4, 4]], dtype=np.float32).T

A_orig_df = pd.DataFrame(A_orig)

A_orig_df #(4 users, 3 movies)

A_df_masked = A_orig_df.copy()
A_df_masked.iloc[0,0]=np.NAN

np_mask = A_df_masked.notnull()
np_mask

# Boolean mask for computing cost only on valid (not missing) entries
tf_mask = tf.Variable(np_mask.values)

A = tf.constant(A_df_masked.values)
shape = A_df_masked.values.shape

#latent factors
rank = 3

# Initializing random H and W
temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
temp_H = np.divide(temp_H, temp_H.max())

temp_W = np.random.randn(shape[0], rank).astype(np.float32)
temp_W = np.divide(temp_W, temp_W.max())

H =  tf.Variable(temp_H)
W = tf.Variable(temp_W)
WH = tf.matmul(W, H)

#cost of Frobenius norm
cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))

# Learning rate
lr = 0.001
# Number of steps
steps = 1000
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()

# Clipping operation. This ensure that W and H learnt are non-negative
clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
clip = tf.group(clip_W, clip_H)

steps = 1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip)
        if i%100==0:
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
