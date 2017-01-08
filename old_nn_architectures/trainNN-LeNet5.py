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


filter1 = 5 # patch_size
filter2 = 2
depth1 = 20
depth2 = 50

hidden = 500 # num of neurons in fully connected hidden layer

num_steps = 64*20
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

    l1_conv = tf.Variable(tf.truncated_normal([filter1, filter1, num_channels, depth1], stddev=0.1))
    b1_conv = tf.Variable(tf.zeros([depth1]))
    l2_conv = tf.Variable(tf.truncated_normal([filter2, filter2, depth1, depth2], stddev=0.1))
    b2_conv = tf.Variable(tf.zeros([depth2]))

    l3 = tf.Variable(tf.truncated_normal([(image_size//4)*(image_size//4)*depth2, hidden], stddev=0.1))
    b3 = tf.Variable(tf.zeros([hidden]))
    l4 = tf.Variable(tf.truncated_normal([hidden, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.zeros([num_labels]))

    weights_and_biases = [l1_conv, b1_conv, l2_conv, b2_conv, l3, b3, l4, b4] # list of all weights and biases

    # computation
    def model(X):
        cl1 = tf.nn.conv2d(input=X,
                           filter=l1_conv,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl1_relu = tf.nn.relu(cl1 + b1_conv)
        cl1_maxpool = tf.nn.max_pool(value=cl1_relu,
                                     ksize=[1,2,2,1], # velikost v jednotlivyh dimenzich: 1 v batch, 2 v height a width, 1 v channels
                                     strides=[1,2,2,1], # posun v jednotlivych dimenzich
                                     padding='SAME')

        cl2 = tf.nn.conv2d(input=cl1_maxpool,
                           filter=l2_conv,
                           strides=[1,1,1,1],
                           padding='SAME')
        cl2_relu = tf.nn.relu(cl2 + b2_conv)
        cl2_maxpool = tf.nn.max_pool(value=cl2_relu,
                                     ksize=[1,2,2,1],
                                     strides=[1,2,2,1],
                                     padding='SAME')

        shape = cl2_maxpool.get_shape().as_list()
        cl2_reshape = tf.reshape(tensor=cl2_maxpool, shape=[shape[0], shape[1]*shape[2]*shape[3]]) # for example from [128, 7, 7, 16] to [16, 784] = [128, 7*7*16]
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
