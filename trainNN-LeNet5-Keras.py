from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.regularizers import l2

import numpy as np
import pickle

# not exactly LeNet5 <---- aded dropout

###############load data###############
pickle_file = './data/fer2013-2.pickle'
f = open(file=pickle_file, mode='rb')
data = pickle.load(file=f)

train_data, train_labels = data['train_data'], data['train_labels']
valid_data, valid_labels = data['valid_data'], data['valid_labels']
test_data, test_labels = data['test_data'], data['test_labels']
del data

##########parameters of data###########
image_size = int(np.sqrt(len(test_data[0])))
num_labels = len(test_labels[0])
num_channels = 1
#######################################


def reformat(dataset, labels): # we have to reshapebecause of conv layers
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels
train_data, train_labels = reformat(train_data, train_labels)
valid_data, valid_labels = reformat(valid_data, valid_labels)
test_data, test_labels = reformat(test_data, test_labels)

valid = valid_data, valid_labels # need to be tuple

print('Loaded from pickle.')
print('Training set:', train_data.shape, ', labels:', train_labels.shape)
print('Validation set:', valid_data.shape, ', labels:', valid_labels.shape)
print('Test set:', test_data.shape, ', labels:', test_labels.shape)
print('#'*40)
############################################################################

###########hyper parameters############
filter1 = 5 # patch_size
filter2 = 2
depth1 = 20
depth2 = 50

hidden = 500 # num of neurons in fully connected hidden layer

batch_size = 64
nb_epoch = 20
#######################################


graph = Sequential()

graph.add(Convolution2D(nb_filter=depth1, nb_row=filter1, nb_col=filter1,
                        input_shape=(image_size, image_size,num_channels),
                        # bacha, v tf je to vetsinou (num_channels,image_size,image_size)
                        border_mode='same', activation='relu',
                        ))
graph.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
graph.add(Convolution2D(nb_filter=depth2, nb_row=filter2, nb_col=filter2,
                        border_mode='same', activation='relu',
                        ))
graph.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
graph.add(Flatten())
graph.add(Dropout(p=0.7))
graph.add(Dense(output_dim=hidden, init='normal', activation='relu',
                        ))
graph.add(Dropout(p=0.7))
graph.add(Dense(output_dim=num_labels, init='normal', activation='softmax',
                        ))

graph.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
graph.fit(x=train_data, y=train_labels, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=valid)

save_file = 'LeNet5.h5'
try:
    graph.save(save_file)
    print('Graph {} successfuly saved.'.format(save_file))
except Exception as e:
    print('Graph wasn\'t saved:', e)

scores = graph.evaluate(x=test_data, y=test_labels)
print('{}: {:.2f}%'.format(graph.metrics_names[1], scores[1]*100))
