import tensorflow as tf
import numpy as np
import pickle
from tools import accuracy

##########parameters of data###########
image_size = 48
num_labels = 7
#######################################

###############load data###############
pickle_file = './data/fer2013.pickle'
f = open(file=pickle_file, mode='rb')
data = pickle.load(file=f)

train_data, train_labels = data['train_data'], data['train_labels']
valid_data, valid_labels = data['valid_data'], data['valid_labels']
test_data, test_labels = data['test_data'], data['test_labels']
del data
print('Loaded from pickle.')
print('Training set:', train_data.shape, ', labels:', train_labels.shape)
print('Validation set:', valid_data.shape, ', labels:', valid_labels.shape)
print('Test set:', test_data.shape, ', labels:', test_labels.shape)
print('#'*40)
#######################################

###########hyper parameters############
batch_size = 128

reg_const = 0.05 # l2 reg constant

# learning rate: exponential decay
initial_learning_rate = 0.05
decay_steps = 2000
decay_rate = 0.75

n_hid1 = 500 # number of neurons in first hidden layer
n_hid2 = 50

num_steps = 128*100
#######################################

graph = tf.Graph()
with graph.as_default():

    # input data
    tf_train_data = tf.placeholder(dtype=tf.float32,
                                   shape=(batch_size, image_size**2))
    tf_train_labels = tf.placeholder(dtype=tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_data = tf.constant(value=valid_data)
    tf_test_data = tf.constant(value=test_data)

    # variables
    global_step = tf.Variable(0) # counts number of steps taken
    w1 = tf.Variable(tf.truncated_normal(shape=[image_size**2, n_hid1], stddev=0.01)) # weights1
    b1 = tf.Variable(tf.zeros(shape=[n_hid1])) # biases1
    w2 = tf.Variable(tf.truncated_normal(shape=[n_hid1, n_hid2], stddev=0.01))
    b2 = tf.Variable(tf.zeros(shape=[n_hid2]))
    w3 = tf.Variable(tf.truncated_normal(shape=[n_hid2, num_labels], stddev=0.01))
    b3 = tf.Variable(tf.zeros(shape=[num_labels]))
    weights_and_biases = [w1, b1, w2, b2, w3, b3] # list of all weights and biases

    # computation
    def model(X, keep_prob = 0.7):
        l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        l1 = tf.nn.dropout(x=l1, keep_prob=keep_prob)
        l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        output = tf.matmul(l2, w3) + b3
        return output

    logits = model(tf_train_data, keep_prob = 0.7) # logits for training

    penalty = 0.5*reg_const*sum([tf.nn.l2_loss(x) for x in weights_and_biases]) # penalties for weights and biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels)) + penalty

    # optimizer
    learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

    # predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(X=valid_data, keep_prob=1))
    test_prediction = tf.nn.softmax(model(X=test_data, keep_prob=1))




with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized\n{}'.format('-'*40))
    for step in range(num_steps):
        offset = (step*batch_size) % (train_labels.shape[0] - batch_size) # we slide through all data
        batch_data = train_data[offset:(offset+batch_size), :]
        batch_labels = train_labels[offset:(offset+batch_size), :]
        feed_dict = {tf_train_data: batch_data, # feed by this batch data
                     tf_train_labels: batch_labels}
        _, l, predictions = session.run(fetches=[optimizer, loss, train_prediction],
                                        feed_dict=feed_dict)

        if (step % 500 == 0):
            print('Minibatch loss at step {}: {:.3f}'.format(step, l))
            print('Learning rate: {:.3f}'.format(learning_rate.eval()))
            print('Minibatch accuracy: {:.1f}%'.format(accuracy(predictions, batch_labels)))
            print('Validation accuracy: {:.1f}%'.format(accuracy(valid_prediction.eval(), valid_labels)))
            print('-'*40)
    print('Test accuracy: {:.1f}%'.format(accuracy(test_prediction.eval(), test_labels)))
