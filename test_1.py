import tensorflow as tf
import matplotlib.pyplot as plt
X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)

hypothesis = X*W

cost = tf.reduce_sum(tf.square(hypothesis-Y))

# Minimize : Gradient Descent using derivative : W -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y) * X)
descent = W - learning_rate * gradient
# W를 업데이트 
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for i in range(-30,50):
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost,W],feed_dict = {W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val,cost_val)
plt.show()