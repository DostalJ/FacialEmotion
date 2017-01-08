import pandas as pd
import numpy as np
import pickle

"""
Prepare data with original labels
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
"""


df = pd.read_csv('./fer2013/fer2013.csv')

l1 = len(df[df['Usage']=='Training'])/len(df)
l2 = len(df[df['Usage']=='PublicTest'])/len(df)
l3 = len(df[df['Usage']=='PrivateTest'])/len(df)
print('Training size:', l1)
print('Validation size:', l2) # we'll use as validation set
print('Test size:', l3)

pixel_depth = 255.0
def split_and_convert(x):
    """from '255 0 127.5' to [1, 0, 0.5]"""
    # convert 255 format format with mean 0 and deviation 0.5
    return [(float(item) - pixel_depth*0.5)/pixel_depth for item in x.split()]
df['pixels'] = df['pixels'].apply(func=split_and_convert) # prevede to vhodnejsiho formatu

num_labels = len(set(df['emotion']))
def reformat(labels):
    """
    from: [0, 2, 1, ...]
    to:
    [[1,0,0,...],
     [0,0,1,...],
     [0,1,0,...],
     ...]
    """
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32) # 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    return labels

train_data = np.array(df[df['Usage']=='Training']['pixels'].values.tolist(), dtype=np.float32) # specific lines of specific column to np.array
train_labels = reformat(df[df['Usage']=='Training']['emotion']) # extract labels and convert to nice format

valid_data = np.array(df[df['Usage']=='PublicTest']['pixels'].values.tolist(), dtype=np.float32) # specific lines of specific column to np.array
valid_labels = reformat(df[df['Usage']=='PublicTest']['emotion']) # extract labels and convert to nice format

test_data = np.array(df[df['Usage']=='PrivateTest']['pixels'].values.tolist(), dtype=np.float32) # specific lines of specific column to np.array
test_labels = reformat(df[df['Usage']=='PrivateTest']['emotion']) # extract labels and convert to nice format

pickle_file = './fer2013-1.pickle'
try:
    f = open(pickle_file, 'wb')
    save = {'train_data': train_data,
            'train_labels': train_labels,
            'valid_data': valid_data,
            'valid_labels': valid_labels,
            'test_data': test_data,
            'test_labels': test_labels}
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
