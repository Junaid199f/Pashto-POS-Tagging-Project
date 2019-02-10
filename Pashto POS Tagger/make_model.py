import numpy as np
import numpy
import pickle, sys, os

import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 50
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
epoch = 5


with open('PickledData/data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)


def generator(all_X, all_y, n_classes, batch_size=BATCH_SIZE):
    num_samples = len(all_X)

    while True:

        for offset in range(0, num_samples, batch_size):
            
            X = all_X[offset:offset+batch_size]
            y = all_y[offset:offset+batch_size]

            y = to_categorical(y, num_classes=n_classes)


            yield shuffle(X, y)


n_tags = len(tag2int)
n=1
names=[]
while(n<=14):
    names.append(int2tag[n])
    n=n+1

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)

# y = to_categorical(y, num_classes=len(tag2int) + 1)

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))

# shuffle the data
X, y = shuffle(X, y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT,random_state=42)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)

# make generators for training and validation
train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_tags + 1)
validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_tags + 1)



with open('PickledData/Glove.pkl', 'rb') as f:
	embeddings_index = pickle.load(f)

print('Total %s word vectors.' % len(embeddings_index))

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(n_tags + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

hist = model.fit_generator(train_generator, 
                     steps_per_epoch=n_train_samples/BATCH_SIZE,
                     validation_data=validation_generator,
                     validation_steps=n_val_samples/BATCH_SIZE,
                     epochs=epoch,
                     verbose=1,
                     workers=1)

if not os.path.exists('Models/'):
    print('MAKING DIRECTORY Models/ to save model file')
    os.makedirs('Models/')

train = True

if train:
    model.save('Models/model.h5')
    print('MODEL SAVED in Models/ as model.h5')
else:
    from keras.models import load_model
    model = load_model('Models/model.h5')

y_test = to_categorical(y_test, num_classes=n_tags+1)
test_results = model.evaluate(X_test, y_test, verbose=0)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))


train_loss=hist.history['loss']
numpy_loss_history = numpy.array(train_loss)
numpy.savetxt("Models/train_loss.txt", numpy_loss_history, delimiter=",")
val_loss=hist.history['val_loss']
numpy_loss_history = numpy.array(val_loss)
numpy.savetxt("Models/validation_loss.txt", numpy_loss_history, delimiter=",")
train_acc=hist.history['acc']
numpy_loss_history = numpy.array(train_acc)
numpy.savetxt("Models/train_accuracy.txt", numpy_loss_history, delimiter=",")
val_acc=hist.history['val_acc']
numpy_loss_history = numpy.array(val_acc)
numpy.savetxt("Models/validation_accuracy.txt", numpy_loss_history, delimiter=",")
xc=range(epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Models/loss.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Models/accuracy.png')


