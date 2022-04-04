#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# save the final model to file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras import layers
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
 
# scores, histories = list(), list()

num_of_cnn_model = 4         # num_of_model define how many model to run with each MNIST, Max = 4
num_of_ann_model = 4         # num_of_model define how many model to run with each MNIST, Max = 4
ep = 40                       # epochs size
bs = 32                      # batch size
layer_activation = 'relu'
output_layer_activation = 'softmax'

cnn_models = list()          # model list to contain each generated model object
ann_models = list()

#load both Digit and Fashion MNIST data
(train_fm_x, train_fm_y), (test_fm_x, test_fm_y) = fashion_mnist.load_data()
(train_dm_x, train_dm_y), (test_dm_x, test_dm_y) = mnist.load_data()


# scale pixels
def prep_cnn_pixels(train, test):
    train_images = train.reshape((60000, 28, 28, 1))
    test_images = test.reshape((10000, 28, 28, 1))
    
    # convert from integers to floats
    # normalize to range 0-1
    train_norm = train_images.astype("float32") / 255
    test_norm = test_images.astype("float32") / 255

    # return normalized images
    return train_norm, test_norm

def prep_ann_pixels(train, test):
    train_images = train.reshape(60000, 784)
    test_images = test.reshape(10000, 784)
    
    # convert from integers to floats
    # normalize to range 0-1
    train_norm = train_images.astype("float32") / 255
    test_norm = test_images.astype("float32") / 255

    # return normalized images
    return train_norm, test_norm



# define cnn model
def define_cnn_model():
    inputs = keras.Input(shape=(28, 28, 1))
    
    for i in range(0,num_of_cnn_model):
        if i == 0:
            # model 1
            x = layers.Conv2D(16,3,activation=layer_activation)(inputs)
            x = layers.Conv2D(32,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(x)
        elif i == 1:
            # model 2
            x = layers.Conv2D(16,3,activation=layer_activation)(inputs)
            x = layers.Conv2D(32,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Conv2D(64,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(x)
        elif i == 2:
            # model 3
            x = layers.Conv2D(16,3,activation=layer_activation)(inputs)
            x = layers.Conv2D(32,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)
            first = layers.Dense(126, activation=layer_activation)(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
        elif i == 3:
            # model 4
            x = layers.Conv2D(16,3,activation=layer_activation)(inputs)
            x = layers.Conv2D(32,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Conv2D(64,3,activation=layer_activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)
            first = layers.Dense(126, activation=layer_activation)(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
    
        cnn_models.append(keras.Model(inputs=inputs, outputs=outputs))
        
    for model in cnn_models:
        #print(model.summary())

        # compile model
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
    
    return cnn_models

def define_ann_model():
    inputs = keras.Input(shape=(784,))
    
    for i in range(0,num_of_ann_model):
        if i == 0:
            # model 1
            x = layers.Dense(128, activation=layer_activation)(inputs)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(24, activation=layer_activation)(x)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(24, activation=layer_activation)(x)
            x = layers.Dropout(0.3)(x)
            first = layers.BatchNormalization()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
        elif i == 1:
            # model 2
            x = layers.Dense(128, activation=layer_activation)(inputs)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(24, activation=layer_activation)(x)
            x = layers.Dropout(0.3)(x)
            first = layers.BatchNormalization()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
        elif i == 2:
            # model 3
            x = layers.Dense(128, activation=layer_activation)(inputs)
            x = layers.Dropout(0.3)(x)
            first = layers.BatchNormalization()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
        elif i == 3:
            # model 4
            x = layers.Dense(128, activation=layer_activation)(inputs)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(24, activation=layer_activation)(inputs)
            first = layers.BatchNormalization()(x)
            outputs = layers.Dense(10,activation=output_layer_activation)(first)
            
        ann_models.append(keras.Model(inputs=inputs, outputs=outputs))
        
    for model in ann_models:
        #print(model.summary())

        # compile model
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
    
    return ann_models


def evaluate_cnn_model(dataX, dataY, tdataX, tdataY, ep = ep, bs = bs):
    # global scores,histories
    lossNAcc = list()
    history = list()

    for model in cnn_models:
        # fit model
        history.append(model.fit(dataX, dataY, epochs = ep, batch_size = bs))
        # evaluate model
        lossNAcc.append(model.evaluate(tdataX, tdataY))

    return history,lossNAcc

def evaluate_ann_model(dataX, dataY, tdataX, tdataY, ep = ep, bs = bs):
    # global scores,histories
    lossNAcc = list()
    history = list()

    for model in ann_models:
        # fit model
        history.append(model.fit(dataX, dataY, epochs = ep, batch_size = bs))
        # evaluate model
        lossNAcc.append(model.evaluate(tdataX, tdataY))

    return history,lossNAcc

def plot_graph(history_list,name,num_model):
    for i in range(0, num_model):
        history = history_list[i]
        pyplot.plot(history.history['accuracy'], label='model'+str(i))
       
    pyplot.title(name)
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.ylim([0.5,1])
    pyplot.legend(loc='lower right')
    pyplot.show()
    
    
def run_test_harness():
    
    train_x_dm, test_x_dm = prep_cnn_pixels(train_dm_x, test_dm_x)
    train_x_fm, test_x_fm = prep_cnn_pixels(train_fm_x, test_fm_x)

    # define model
    #print('Model summary: \n')
    define_cnn_model()
    
    # evaluate each cnn model
    print('\n', 'Evaluate Digit MNIST model: \n')
    cnn_dm_history, cnn_dm_lossNAcc = evaluate_cnn_model(train_x_dm, train_dm_y, test_x_dm, test_dm_y)
    print('\n','Evaluate Fashion MNIST model: \n')
    cnn_fm_history, cnn_fm_lossNAcc = evaluate_cnn_model(train_x_fm, train_fm_y, test_x_fm, test_fm_y)
    
    train_x_dm, test_x_dm = prep_ann_pixels(train_dm_x, test_dm_x)
    train_x_fm, test_x_fm = prep_ann_pixels(train_fm_x, test_fm_x)
    
    define_ann_model()
    
    # evaluate each ann model
    print('\n', 'Evaluate Digit MNIST model: \n')
    ann_dm_history, ann_dm_lossNAcc = evaluate_ann_model(train_x_dm, train_dm_y, test_x_dm, test_dm_y)
    print('\n','Evaluate Fashion MNIST model: \n')
    ann_fm_history, ann_fm_lossNAcc = evaluate_ann_model(train_x_fm, train_fm_y, test_x_fm, test_fm_y)

    
    plot_graph(cnn_dm_history,"mnist dataset",num_of_cnn_model)
   
    plot_graph(cnn_fm_history,"fashion dataset",num_of_cnn_model)
    
    for i in range(0, num_of_cnn_model):
        print('\n', 'Model ', i, ' :\n')
        _, acc = cnn_dm_lossNAcc[i]
        print('> CNN accuracy on mnist dataset %.3f\n' % (acc * 100.0))
        _, acc = cnn_fm_lossNAcc[i]
        print('> CNN accuracy on fashion mnist dataset %.3f\n' % (acc * 100.0))
        
    
    plot_graph(ann_dm_history,"mnist dataset",num_of_ann_model)
   
    plot_graph(ann_fm_history,"fashion dataset",num_of_ann_model)
    
    for i in range(0, num_of_ann_model):
        print('\n', 'Model ', i, ' :\n')
        _, acc = ann_dm_lossNAcc[i]
        print('> ANN accuracy on mnist dataset %.3f\n' % (acc * 100.0))
        _, acc = ann_fm_lossNAcc[i]
        print('> ANN accuracy on fashion mnist dataset %.3f\n' % (acc * 100.0))


# entry point, run the test harness
run_test_harness()