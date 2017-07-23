import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

cast = tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32)
sess = tf.Session()

print(sess.run(cast))