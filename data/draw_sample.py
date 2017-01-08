import pickle
import numpy as np
import matplotlib.pyplot as plt


pickle_file = './data/fer2013.pickle'
f = open(file=pickle_file, mode='rb')
data = pickle.load(file=f)

train_data, train_labels = data['train_data'], data['train_labels']
valid_data, valid_labels = data['valid_data'], data['valid_labels']
test_data, test_labels = data['test_data'], data['test_labels']
del data



"""
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
"""
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



image_size = int(np.sqrt(train_data.shape[1])) # number of pixels
num_images = int(train_data.shape[0])
def draw(output='show', filepath='./examples/figure', cmap=None, colorbar=True, axis=True):
    """
    Draw random image from training dataset.
    ________________
    Parameters:
    output: 'show'/'save'
    filepath: path to save image
    cmap = 'gray' (for grayscale image)
    colorbar: True/False (show/hide colorbar)
    axis: True/False (show/hide axis)
    """
    pic_num = np.random.randint(low=0, high=num_images) # randomly choose image
    image = np.reshape(train_data[pic_num], newshape=(image_size,image_size)) # convert to usable format (from one big 1D array of puxels to 2D array of pixels)
    plt.imshow(X=image, cmap=cmap) # draw
    plt.title(emotions[np.argmax(train_labels[pic_num])]) # title with emotion
    if not(axis):
        plt.axis('off')
    if colorbar:
        plt.colorbar() # show colorbar

    if output == 'show':
        plt.show()
    elif output == 'save':
        try:
            plt.savefig(filepath)
        except Exception as e:
            print("Can't save picture '{}':".format(filepath), e)
        plt.clf()
    else:
        raise Exception('Wrong output format!!!')
# draw()


def make_examples():
    """makes examples and saves them to ./examples"""
    for i in range(2,11):
        draw(output='save',
             filepath='./examples/figure_{}'.format(i),
             cmap='gray')
