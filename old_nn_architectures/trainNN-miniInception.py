import tensorflow as tf
import numpy as np
import pickle
from tools import accuracy

##########parameters of data###########
image_size = 48
num_labels = 7
num_channels = 1
#######################################

###############load data###############
pickle_file = './data/fer2013.pickle'
f = open(file=pickle_file, mode='rb')
data = pickle.load(file=f)

train_data, train_labels = data['train_data'], data['train_labels']
valid_data, valid_labels = data['valid_data'], data['valid_labels']
test_data, test_labels = data['test_data'], data['test_labels']
del data

def reformat(dataset, labels): # we have to reshapebecause of conv layers
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels
train_data, train_labels = reformat(train_data, train_labels)
valid_data, valid_labels = reformat(valid_data, valid_labels)
test_data, test_labels = reformat(test_data, test_labels)

print('Loaded from pickle.')
print('Training set:', train_data.shape, ', labels:', train_labels.shape)
print('Validation set:', valid_data.shape, ', labels:', valid_labels.shape)
print('Test set:', test_data.shape, ', labels:', test_labels.shape)
print('#'*40)
#######################################

###########hyper parameters############
batch_size = 64

reg_const = 0.05 # l2 reg constant

# learning rate: exponential decay
initial_learning_rate = 0.04
decay_steps = 1000
decay_rate = 0.65


filter1 = 3 # patch_size
filter_inc = 1
filter2b = 3
filter2c = 5

depth1 = 192
depth2a = 64
depth2b_inc = 96
depth2b = 128
depth2c_inc = 16
depth2c = 32
depth2d_inc = 32

hidden3 = 500 # num of neurons in fully connected hidden layer

num_steps = 4001
#######################################

graph = tf.Graph()
with graph.as_default():

    # input data
    tf_train_data = tf.placeholder(shape=(batch_size, image_size, image_size, num_channels),
                                   dtype=tf.float32)
    tf_train_labels = tf.placeholder(shape=(batch_size, num_labels),
                                     dtype=tf.float32)
    tf_valid_data = tf.constant(value=valid_data)
    tf_test_data = tf.constant(value=test_data)

    # variables
    global_step = tf.Variable(0) # counts number of steps taken

    l1 = tf.Variable(tf.truncated_normal([filter1, filter1, num_channels, depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([depth1]))

    l2a_inc = tf.Variable(tf.truncated_normal([filter_inc, filter_inc, depth1, depth2a], stddev=0.1))
    b2a_inc = tf.Variable(tf.zeros([depth2a]))

    l2b_inc = tf.Variable(tf.truncated_normal([filter_inc, filter_inc, depth1, depth2b_inc], stddev=0.1))
    b2b_inc = tf.Variable(tf.zeros([depth2b_inc]))
    l2b = tf.Variable(tf.truncated_normal([filter2b, filter2b, depth2b_inc, depth2b], stddev=0.1))
    b2b = tf.Variable(tf.zeros([depth2b]))

    l2c_inc = tf.Variable(tf.truncated_normal([filter_inc, filter_inc, depth1, depth2c_inc], stddev=0.1))
    b2c_inc = tf.Variable(tf.zeros([depth2c_inc]))
    l2c = tf.Variable(tf.truncated_normal([filter2c, filter2c, depth2c_inc, depth2c], stddev=0.1))
    b2c = tf.Variable(tf.zeros([depth2c]))

    l2d_inc = tf.Variable(tf.truncated_normal([filter_inc, filter_inc, depth1, depth2d_inc], stddev=0.1))
    b2d_inc = tf.Variable(tf.zeros([depth2d_inc]))

# TODO: 147456 se tu prikouzlilo --> udelat poradne
    l3 = tf.Variable(tf.truncated_normal([147456, hidden3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([hidden3]))

    l4 = tf.Variable(tf.truncated_normal([hidden3, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.zeros([num_labels]))

    weights_and_biases = [] # list of all weights and biases

    # computation
    def model(X):
        cl1 = tf.nn.conv2d(input=X,
                           filter=l1,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl1_relu = tf.nn.relu(cl1 + b1)
        cl1_maxpool = tf.nn.max_pool(value=cl1_relu,
                                     ksize=[1,3,3,1], # velikost v jednotlivyh dimenzich: 1 v batch, 2 v height a width, 1 v channels
                                     strides=[1,2,2,1], # posun v jednotlivych dimenzich
                                     padding='SAME')


        # a = inception
        # b = inception + 3x3
        # c = inception + 5x5
        # d = maxpool 3x3 + inception
        cl2a_inc = tf.nn.conv2d(input=cl1_maxpool,
                           filter=l2a_inc,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2a_inc_relu = tf.nn.relu(cl2a_inc + b2a_inc)

        cl2b_inc = tf.nn.conv2d(input=cl1_maxpool,
                           filter=l2b_inc,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2b_inc_relu = tf.nn.relu(cl2b_inc + b2b_inc)
        cl2b = tf.nn.conv2d(input=cl2b_inc_relu,
                           filter=l2b,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2b_relu = tf.nn.relu(cl2b + b2b)

        cl2c_inc = tf.nn.conv2d(input=cl1_maxpool,
                           filter=l2c_inc,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2c_inc_relu = tf.nn.relu(cl2c_inc + b2c_inc)
        cl2c = tf.nn.conv2d(input=cl2c_inc_relu,
                           filter=l2c,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2c_relu = tf.nn.relu(cl2c + b2c)

        cl2d_maxpool = tf.nn.max_pool(value=cl1_maxpool,
                                     ksize=[1,3,3,1],
                                     strides=[1,1,1,1],
                                     padding='SAME')
        cl2d_inc = tf.nn.conv2d(input=cl2d_maxpool,
                                filter=l2d_inc,
                                strides=[1,1,1,1],
                                padding='SAME')
        cl2d_inc_relu = tf.nn.relu(cl2d_inc + b2d_inc)

        # print(cl2a_inc_relu.get_shape().as_list())
        # print(cl2b_relu.get_shape().as_list())
        # print(cl2c_relu.get_shape().as_list())
        # print(cl2d_inc_relu.get_shape().as_list())

        cl2 = tf.concat(concat_dim=3, values=[cl2a_inc_relu, cl2b_relu, cl2c_relu, cl2d_inc_relu])
        # print(cl2.get_shape().as_list())
        shape = cl2.get_shape().as_list()
        cl2_reshape = tf.reshape(tensor=cl2, shape=[shape[0], shape[1]*shape[2]*shape[3]])

        cl3 = tf.nn.relu(tf.matmul(cl2_reshape, l3) + b3)
        output = tf.matmul(cl3, l4) + b4
        return output

    logits = model(tf_train_data) # logits for training

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
    valid_prediction = tf.nn.softmax(model(X=valid_data))
    test_prediction = tf.nn.softmax(model(X=test_data))




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
