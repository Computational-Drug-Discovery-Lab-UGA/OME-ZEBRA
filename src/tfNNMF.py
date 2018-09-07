import tensorflow as tf
import numpy as np
import pandas as pd
print("from tfnnmf.py - all modules loaded")

def tensorflowNNMF(A_orig, rank, iterations):
    print("Starting tensorflowNNMF nnmf prep")

    A_orig_df = pd.DataFrame(A_orig)

    shape = A_orig_df.values.shape
    print(shape)
    print(rank)

    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())
    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H =  tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    #causing a seg fault when trying to access values from data frame
    A = tf.constant(A_orig_df.values)

    #cost of Frobenius norm
    cost = tf.reduce_mean(tf.pow(A - WH, 2))

    # Learning rate
    lr = 1.0
    train_step = tf.train.AdamOptimizer(lr).minimize(cost)

    # Clipping operation. This ensure that W and H learnt are non-negative
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)

    print("Starting tensorflowNNMF")
    
    previousLoss = 99999.999
    lossThresh = 1.0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            sess.run(train_step)
            sess.run(clip)
            if i%10==0:
                print("\nCost: %f" % sess.run(cost))
                print("*"*40)
            if i == 0:
                previousLoss = sess.run(cost)
            else:
                if (previousLoss - sess.run(cost)) < lossThresh:
                    lr = lr * 0.1
                    train_step = tf.train.AdamOptimizer(lr).minimize(cost)
                    lossThresh = lossThresh * 0.5
                
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)

    print("tensorflow nnmf has completed")
    return learnt_W, learnt_H
