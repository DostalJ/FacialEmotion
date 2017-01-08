import cv2
from keras import

"""
Prepare data with 'disgust' included in 'anger' --> only 6 labels
0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
"""


def predict():
    """Makes prediction about emotions on the picture"""
    pass


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

write = False
i = 0
while True:
    _, image = cap.read()

    k1 = cv2.waitKey(5) & 0xFF
    if k1 == 13:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        face = cv2.resize(image, (48, 48))
        # TODO: classify the face

        i += 1

        write = True

    if write:
        cv2.rectangle(img=image, color=(255,0,0),
                      pt1=(0,0,), pt2=(200,100),
                      thickness=-1,)
        cv2.putText(img=image, org=(100,55), # sloupce, radky
                    text='Number: {}'.format(i),
                    fontScale=1, fontFace=6,
                    thickness=2, color=(0,255,0))

    cv2.imshow('im', image)

    k2 = cv2.waitKey(30) & 0xFF
    if k1 == 27: # 27 = esc
        break

cap.release()
cv2.destroyAllWindows()
