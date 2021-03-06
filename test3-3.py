import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)

hypothesis = X*W

gradient = tf.reduce_mean((W*X - Y)*X) * 2
#gradient는 직접 기울기를 계산한 값

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost, [W])
#gvs는 optimizer의 기울기를 optimier.compute_gradients함수를 이용해 직접 구한 것
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
