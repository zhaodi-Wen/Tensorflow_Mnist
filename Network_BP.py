import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import os

# 屏蔽waring信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""------------------加载数据---------------------"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""------------------构建模型---------------------"""
# 定义输入、输出、权重、偏置
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b  # 预测值
y_ = tf.placeholder(tf.float32, [None, 10])  # 真实值

# 定义误差函数为交叉熵:计算预测值和真实值之间的偏差，取平均
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# 定义训练方式为SGD随机梯度下降方法:学习速率=0.5，误差函数就是上面定义的交叉熵
train_op=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""------------------训练模型----------------------"""
# 创建会话来进行训练
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

# 训练模型(随机训练）:每次随机取100张图片去更新一遍网络，总共1000次
for i in range(1000):
    batch_X,batch_Y=mnist.train.next_batch(100)
    sess.run(train_op,feed_dict={x:batch_X,y_:batch_Y})

"""------------------评估模型-----------------------"""
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))