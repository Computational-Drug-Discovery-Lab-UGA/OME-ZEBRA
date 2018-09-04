import tensorflow as tf
import numpy as np
import pandas as pd

def tensorNNMF(A_orig, rank):
    A_orig = A_orig.astype(np.float32)
    print("numpy V matrix is loaded")

    A_orig_df = pd.DataFrame(A_orig)
    print("V Array created")

    A = tf.constant(A_orig_df.values)
    shape = A_orig_df.values.shape

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

    return learnt_W, learnt_H
