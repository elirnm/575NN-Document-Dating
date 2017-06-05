"""modeled on the system described in
Zhang, Xiang, Junbo Zhao, and Yann LeCun. 2015. “Character-Level
    Convolutional Networks for Text Classification.” In "Advances in
    Neural Information Processing Systems" 28, edited by C. Cortes,
    N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, 649–657.
    Curran Associates, Inc. 
    http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf.

with some small changes
"""
 

import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from data import ALPHABET, DATA_DIR

def get_label(doc):
    lab = doc.corpus_section[:2]
    if lab[1] == "X" and len(doc.corpus_section) == 4:
        lab = lab[0] + doc.corpus_section[3]
    return lab.strip()
    
with open(DATA_DIR + "test_docs.pkl", 'rb') as f:
    test_docs = pickle.load(f)
with open(DATA_DIR + "devtest_docs.pkl", 'rb') as f:
    devtest_docs = pickle.load(f)
with open(DATA_DIR + "train_docs.pkl", 'rb') as f:
    train_docs = pickle.load(f)
    
# set parameters:
batch_size = 128
epochs = 10
maxchars = 1014

def make_onehot(char):
    vec = [0 for x in range(len(ALPHABET))]
    vec[ALPHABET.index(char)] = 1
    return vec
    
# convert text to one-hot character matrices
x_train = []
y_train = []
for doc in train_docs:
    int_chars = [make_onehot(doc.chars[i]) for i in range(maxchars) if i < len(doc.chars)]
    lab = get_label(doc)
    x_train.append(int_chars)
    y_train.append(lab)
    
x_devtest = []
y_devtest = []
for doc in devtest_docs:
    int_chars = [make_onehot(doc.chars[i]) for i in range(maxchars) if i < len(doc.chars)]
    lab = get_label(doc)
    x_devtest.append(int_chars)
    y_devtest.append(lab)
    
x_test = []
y_test = []
for doc in test_docs:
    int_chars = [make_onehot(doc.chars[i]) for i in range(maxchars) if i < len(doc.chars)]
    lab = get_label(doc)
    x_test.append(int_chars)
    y_test.append(lab)
    
# pad the matrices just in case
x_train = sequence.pad_sequences(x_train, maxlen=maxchars)
x_devtest = sequence.pad_sequences(x_devtest, maxlen=maxchars)
x_test = sequence.pad_sequences(x_test, maxlen=maxchars)

# convert labels to integers
label_vals = {}
val_labels = {}
for i, label in enumerate(set(y_train + y_devtest + y_test)):
    val_labels[i] = label
    label_vals[label] = i
y_train = [label_vals[x] for x in y_train]
y_devtest = [label_vals[x] for x in y_devtest]
y_test = [label_vals[x] for x in y_test]

# convert labels to categorical
num_classes = len(set(y_train + y_devtest + y_test))
y_train = to_categorical(y_train, num_classes)
y_devtest = to_categorical(y_devtest, num_classes)
y_test = to_categorical(y_test, num_classes)


# build the model
model = Sequential()
model.add(Conv1D(256, 7, input_shape=(maxchars, len(ALPHABET),)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(256, 7))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(256, 3))
model.add(Activation('relu'))

model.add(Conv1D(256, 3))
model.add(Activation('relu'))

model.add(Conv1D(256, 3))
model.add(Activation('relu'))

model.add(Conv1D(256, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

model.add(GlobalMaxPooling1D())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
result = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_split=0.1)

score = model.evaluate(x_devtest, y_devtest,
                       batch_size=batch_size)
print('\nDevTest loss:', score[0])
print('DevTest accuracy:', score[1])

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size)
fi = open("res", 'w')
fi.write('Test loss:' + str(score[0]) + "\n")
fi.write('Test accuracy:' + str(score[1]) + "\n")
