from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import pickle

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
valid = valid_data, valid_labels # need to be tuple
#######################################

###########hyper parameters############
batch_size = 128

reg_const = 0.05 # l2 reg constant

n_hid1 = 500 # number of neurons in first hidden layer
n_hid2 = 50

num_epochs = 100
#######################################

graph = Sequential()
graph.add(Dense(output_dim=n_hid1,
                input_dim=image_size**2,
                init='normal',
                activation='relu'))
graph.add(Dropout(p=0.7))
graph.add(Dense(output_dim=n_hid2,
                init='normal',
                activation='relu'))
graph.add(Dense(output_dim=num_labels,
                init='normal',
                activation='softmax'))

# sgd je rychlejsi, ale ma horsi performance
graph.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
graph.fit(x=train_data, y=train_labels,
          validation_data=valid,
          batch_size=batch_size, nb_epoch=num_epochs)

scores = graph.evaluate(x=test_data, y=test_labels)
print('{}: {:.2f}%'.format(graph.metrics_names[1], scores[1]*100))
