# Facial Emotion Recognition
Because, who understand women?
#### Data
We are using data from public [Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
They contain about 200 MB of raw picture data in 48x48 pixel format. All images are labeled with one of the 7 labels:

| Label    | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Emotion   | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral |

Because there is only few pictures of emotion 'disgust', we've merged this emotion with 'anger'. Similarly to this [project](https://github.com/JostineHo/mememoji/blob/master/src/fer2013datagen.py).
| Label    | 0 | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Emotion   | Anger | Fear | Happy | Sad | Surprise | Neutral |
