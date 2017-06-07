import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from data import *

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
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
max_words = 10000
tok = Tokenizer()

# convert text to binary matrices
tok.fit_on_texts([x.cleaned_text for x in train_docs])
x_train = tok.texts_to_matrix([x.cleaned_text for x in train_docs])
y_train = [get_label(x) for x in train_docs]

x_devtest = tok.texts_to_matrix([x.cleaned_text for x in devtest_docs])
y_devtest = [get_label(x) for x in devtest_docs]

x_test = tok.texts_to_matrix([x.cleaned_text for x in test_docs])
y_test = [get_label(x) for x in test_docs]

# pad the matrices just in case
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_devtest = sequence.pad_sequences(x_devtest, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

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
model.add(Dense(512, input_shape=(max_words,)))
# model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
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
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
