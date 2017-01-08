from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

"""
Prepare data with 'disgust' included in 'anger' --> only 6 labels
0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
"""

pixel_depth = 255.0
num_channels = 1
image_size = 48

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# f = open('./data/fer2013-2.pickle', 'rb')
# data = pickle.load(f)
# test_data, test_labels = data['test_data'], data['test_labels']
# del data
# def reformat(dataset, labels): # we have to reshapebecause of conv layers
#     dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
#     return dataset, labels
# test_data, test_labels = reformat(test_data, test_labels)

graph = load_model(filepath='LeNet5.h5')

# indexes = np.random.randint(low=0, high=test_labels.shape[0], size=20)
# for index in indexes:
#     plt.imshow(test_data[index,:,:,0], cmap='gray')
#     emotions = graph.predict(test_data[index,:,:,:].reshape(1,image_size,image_size,num_channels))[0]
#     plt.title("Angry: {:.2f}\nFear: {:.2f}\nHappy: {:.2f}\nSad: {:.2f}\nSurprise: {:.2f}\nNeutral: {:.2f}\nLabel: {}".format(emotions[0],
#                                                                                                                             emotions[1],
#                                                                                                                             emotions[2],
#                                                                                                                             emotions[3],
#                                                                                                                             emotions[4],
#                                                                                                                             emotions[5],
#                                                                                                                             np.argmax(test_labels[index])))
#     plt.show()

image = cv2.imread('surprise.jpg')

reformater = np.empty((48,48))
reformater.fill(pixel_depth*0.2)

def predict(image):
    """Makes prediction about emotions on the picture"""
    # newshape = (batch_size, image_size, image_size, num_channels)
    image = (image - reformater)/pixel_depth
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()
    image = np.reshape(a=image, newshape=(1, image_size, image_size, num_channels)).astype(np.float32)
    emotions = graph.predict(image)[0]
    return emotions

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)#, 1.3, 5)
if len(faces) != 0:
    (x,y,w,h) = faces[0]
    # TODO: casem mohu smazat
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    face = cv2.resize(gray[y: y+h, x: x+w], (image_size, image_size))
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(face, cmap='gray')
    plt.colorbar()
    plt.show()
    plt.clf()
    emotions = predict(face)

    print("Angry: {:.2f}\nFear: {:.2f}\nHappy: {:.2f}\nSad: {:.2f}\nSurprise: {:.2f}\nNeutral: {:.2f}".format(emotions[0],
                                                                                                          emotions[1],
                                                                                                          emotions[2],
                                                                                                          emotions[3],
                                                                                                          emotions[4],
                                                                                                          emotions[5],))
