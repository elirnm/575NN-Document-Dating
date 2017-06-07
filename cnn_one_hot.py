import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from data import ALPHABET, CASE_I_ALPHABET, DATA_DIR

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
batch_size = 32
kernel_size = 10
epochs = 10
maxchars = 256

def make_onehot(char):
    vec = [0 for x in range(len(CASE_I_ALPHABET))]
    vec[CASE_I_ALPHABET.index(char)] = 1
    return vec
    
# convert text to one-hot character matrices
x_train = []
y_train = []
for doc in train_docs:
    int_chars = [make_onehot(doc.chars[i].lower()) for i in range(maxchars) if i < len(doc.chars)]
    lab = get_label(doc)
    x_train.append(int_chars)
    y_train.append(lab)
    
x_devtest = []
y_devtest = []
for doc in devtest_docs:
    int_chars = [make_onehot(doc.chars[i].lower()) for i in range(maxchars) if i < len(doc.chars)]
    lab = get_label(doc)
    x_devtest.append(int_chars)
    y_devtest.append(lab)
    
x_test = []
y_test = []
for doc in test_docs:
    int_chars = [make_onehot(doc.chars[i].lower()) for i in range(maxchars) if i < len(doc.chars)]
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
model.add(Conv1D(250, kernel_size, input_shape=(maxchars, len(CASE_I_ALPHABET),)))

# model.add(Conv1D(250, 3))
model.add(Activation('relu'))
# model.add(MaxPooling1D())

model.add(Conv1D(250, kernel_size))
model.add(Activation('relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(512))
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
print('Test loss:' + str(score[0]) + "\n")
print('Test accuracy:' + str(score[1]) + "\n")
