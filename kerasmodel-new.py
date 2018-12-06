import numpy as np
import keras
import time
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import model_from_json
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#convert vector to binary
print('converting vectors to binary')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('done converting')
#normalize training data from 0-255 to 0-1.0 (add noise)
print('normalizing training data')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#-- ADD NOISE TO TRAINING DATA HERE --#
x_train = x_train / 255.0
x_test = x_test / 255.0

#-- will add random noise to entire image --#
def addgaussian(images):
    print('entering noise adder')
    noisy_images = images
    noise = np.random.normal(loc = 0.0, scale = .125, size = (32,32,3))
    counter = 0
    for i in range (len(noisy_images)):
        odds = np.random.randint(1,11)
        if odds == 10:
            #print('adding noise to image', i)
            noisy_images[i] = images[i] + noise
            counter = counter + 1
            #cap noisy images to 0 <= x <= 255
            for a in range(32):
                for j in range(32):
                    for k in range(3):
                        if noisy_images[i][a][j][k] < 0:
                            noisy_images[i][a][j][k] = 0
                        if noisy_images[i][a][j][k] > 255:
                            noisy_images[i][a][j][k] = 255
    counter = counter / 500
    print('Percent of noisy data = ', counter, "%")
    return noisy_images


def examplegaus(images, index):
    noise = np.random.normal(loc = 0.0, scale = .125, size = (32,32,3))
    noisy_image = images[index] + noise
    for a in range(32):
                for j in range(32):
                    for k in range(3):
                        if noisy_image[a][j][k] < 0:
                            noisy_image[a][j][k] = 0
                        if noisy_image[a][j][k] > 255:
                            noisy_image[a][j][k] = 255
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(images[index], interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image, interpolation='none')
    plt.show()

#-- grayscale the data to make training faster --#
def grayscale(images):
    #using luma coding average
    types = 'float32'
    r, g, b = np.asarray(.3, dtype=types), np.asarray(.59, dtype=types), np.asarray(.11, dtype=types)
    grayed = r * images[:, :, :, 0] + g * images[:, :, :, 1] + b * images[:, :, :, 2]
    grayed = np.expand_dims(grayed, axis=3)
    return grayed

# display a randomly chosen image
def showimage(index):
    img = index
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(x_train[img], interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(x_train_new[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
    plt.show()

print("grayscaling images...")
#examplegaus(x_train, np.random.randint(0,40000))
x_train_new = addgaussian(x_train)
print('grayscaling training data...')
x_train_new = grayscale(x_train_new)
print('grayscaling testing data...')
x_test_new  = grayscale(x_test)
# showimage(64)


#split data into training and validation arrays
print("splitting data into trainng and testing")
x_train_new, x_val_new, y_train, y_val = train_test_split(x_train_new, y_train, test_size=0.2, random_state=0)
print('X_train_new shape:', x_train_new.shape)
print('X_val_new shape:', x_val_new.shape)


# define constants
batch_size = 64
epoch_max = 60
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0)

def fit(model):
    hist = model.fit(x_train_new, y_train,
                    nb_epoch=epoch_max,
                    batch_size=batch_size, 
                    validation_data=(x_val_new, y_val), 
                    callbacks=[early_stop], 
                    shuffle=True, verbose=0)
    return hist

def evaluate(model, hist, plt_path):
    score = model.evaluate(x_test_new, y_test, verbose=0)
    print('Test loss: %.3f' % score[0])
    print('Test accuracy: %.3f' % score[1])
    plot_validation_history(hist, plt_path)

def plot_validation_history(history, imagepath):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc = 'upper left')
    plt.savefig(imagepath)
    plt.show()


nb_classes = 10
imgsize = 32
imgchannels = 1
model = Sequential()

#-- generate random images --#
createrandimg = ImageDataGenerator(rotation_range = 20,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             shear_range = 0.1,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'nearest')

# # visualize some generated images
# print('creating random images...')
# plt.figure(figsize=(6, 6))
# (x_batch, y_batch) = createrandimg.flow(x_train_new, y_train, batch_size = 9).next()
# for i in range(9):
#     plt.subplot(3, 3, (i + 1))
#     plt.imshow(x_batch[i, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
# plt.show()
 
# print('y_batch shape:', x_batch.shape)
# print('y_batch shape:', y_batch.shape)
drop_rate = 0.2

# convolutional hidden layers
for i in range(6):
    model.add(Conv2D(64, (3,3), input_shape = (imgsize, imgsize, imgchannels), 
                     padding = 'same', activation = 'relu',
                     ))
    model.add(Dropout(drop_rate))
    if (i + 1) % 2 == 0:
        model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same',
                               strides=None))
    
print('Output shape of last concolution layers: {0}'.format(model.output_shape))
model.add(Flatten())


# fully connected hidden layers
for i in range(2):
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    
# output layer
model.add(Dense(nb_classes, activation = 'softmax'))

# compile model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# training
print("Now training...")
starttime = time.time()
hist = model.fit_generator(
                     createrandimg.flow(x_train_new, y_train, batch_size=batch_size),
                     steps_per_epoch = x_train_new.shape[0]/batch_size,
                     epochs = epoch_max,
                     validation_data = (x_val_new, y_val),
                     callbacks = [early_stop],
                     verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 


# Plot training & validation accuracy values
print("--- training took %s seconds ---" % (time.time() - starttime))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#display model
#plot_model(model, to_file='modelplot.png')
# evaluation
#evaluate(model, hist, 'output/fig-val-loss.png')