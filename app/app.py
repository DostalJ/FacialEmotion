import cv2
from keras.models import load_model
import numpy as np

"""
Prepare data with 'disgust' included in 'anger' --> only 6 labels
0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
"""




pixel_depth = 255.0
num_channels = 1
image_size = 48

reformater = np.empty((48,48))
reformater.fill(pixel_depth*0.5)


graph = load_model(filepath='LeNet5.h5')



def predict(image):
    """Makes prediction about emotions on the picture"""
    # newshape = (batch_size, image_size, image_size, num_channels)
    image = (image - reformater)/pixel_depth
    image = np.reshape(a=image, newshape=(1, image_size, image_size, num_channels)).astype(np.float32)
    emotions = graph.predict(image)[0]
    return emotions
def write_on_image(emotions):
    print('Angry:', emotions[0])
    print('Fear:', emotions[1])
    print('Happy:', emotions[2])
    print('Sad:', emotions[3])
    print('Surprise:', emotions[4])
    print('Neutral:', emotions[5])
    print('-'*40)

    cv2.putText(img=image, org=(50,50), # sloupce, radky
                text='Angry: {:2f}%'.format(emotions[0]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))
    cv2.putText(img=image, org=(50,100), # sloupce, radky
                text='Fear: {:.2f}%'.format(emotions[1]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))
    cv2.putText(img=image, org=(50,150), # sloupce, radky
                text='Happy: {:.2f}%'.format(emotions[2]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))
    cv2.putText(img=image, org=(50,200), # sloupce, radky
                text='Sad: {:.2f}%'.format(emotions[3]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))
    cv2.putText(img=image, org=(50,250), # sloupce, radky
                text='Surprise: {:.2f}%'.format(emotions[4]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))
    cv2.putText(img=image, org=(50,300), # sloupce, radky
                text='Neutral: {:.2f}%'.format(emotions[5]*100),
                fontScale=1, fontFace=6,
                thickness=2, color=(0,255,0))


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

write = False
i = 0
while True:
    _, image = cap.read()

    # k1 = cv2.waitKey(5) & 0xFF
    # if k1 == 13:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)#, 1.3, 5)
    if len(faces) != 0:
        (x,y,w,h) = faces[0]
        # TODO: casem mohu smazat
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        face = cv2.resize(gray[y: y+h, x: x+w], (image_size, image_size))
        emotions = predict(face)

        write_on_image(emotions=emotions)

    cv2.imshow('im', image)

    k2 = cv2.waitKey(30) & 0xFF
    if k2 == 27: # 27 = esc
        break

cap.release()
cv2.destroyAllWindows()
