import tensorflow as tf

onehot = tf.one_hot([[0],[1],[2],[0]], depth = 3)

sess = tf.Session()

print(sess.run(onehot))


