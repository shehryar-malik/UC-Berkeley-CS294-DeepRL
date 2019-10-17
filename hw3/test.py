import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def _double(a):
    return tf.cast(2*a, tf.float32)

def see():
    a = np.array([1.2,2.3,3.5,4.2,5.9])
    b = [1.1,2.5,3.6,3.9,5.2]
    c = [10000  ,30000  ,50000  ,60000  ,150000]

    for i in range(4):
        a = [2*aa for aa in a]
        b = [2*bb for bb in b]
        if not plt.get_fignums():
            fig = plt.figure()

        t = [t_/c[0] for t_ in c]
        print(a)
        print(t)
        plt.plot(t, a, label="mean reward")
        plt.plot(t, b, label="best mean reward")
        
        plt.xlabel("x%d iteration" % c[0] )
        fig.legend(loc="best")

if __name__=="__main__":
    """
    n = 4
    
    ph1 = tf.placeholder(tf.float32, [None,n])
    ph2 = tf.placeholder(tf.int32, [None])
    mx1 = 1-ph2

    with tf.Session() as sess:
        in1 = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.]])
        in2 = np.array([1,0,1,1,1,0])
        ou1 = sess.run(mx1, {ph1: in1, ph2: in2})
        print(ou1)
    """

    see()
    fig = plt.gcf()
    plt.show()
