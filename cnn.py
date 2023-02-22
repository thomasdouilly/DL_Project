import keras
from keras.models import Sequential
from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, ZeroPadding2D

import data_import
import utils
import numpy as np

def get_cnn_structure():
    
    print('***** Creation of a CNN *****')
    clf = Sequential()

    clf.add(ZeroPadding2D(padding = 1, input_shape = (224,224,3)))
    clf.add(Conv2D(64, kernel_size = 3, strides = 1, activation='relu', name = "conv_1_1"))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(64, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(MaxPooling2D(pool_size = 2, strides = 2))

    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(128, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(128, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(MaxPooling2D(pool_size = 2, strides = 2))

    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(256, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(256, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(256, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(MaxPooling2D(pool_size = 2, strides = 2))

    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(MaxPooling2D(pool_size = 2, strides = 2))

    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(Conv2D(512, kernel_size = 3, strides = 1, activation='relu'))
    clf.add(ZeroPadding2D(padding = 1))
    clf.add(GlobalMaxPooling2D())
    
    loss = keras.losses.MeanAbsolutePercentageError()
    
    clf.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.1), loss = loss)
    
    print('***** CNN created *****')
    
    return clf

def train_cnn(model, training_imgs_for_cnn, training_features_for_cnn):
    print('***** Training of the CNN *****')
    model.fit(training_imgs_for_cnn, training_features_for_cnn, epochs = 2, batch_size = 64)
    print('***** Training completed *****')
    
    return model

def cnn_learning(model, number_of_batch, batch_size, training_ids, training_features):
    
    for i in range(number_of_batch):
        
        print("********** Beginning of batch n°{} **********".format(i+1))
        
        training_imgs_for_cnn, training_features_for_cnn = data_import.get_training_data_for_cnn(batch_size, training_ids, training_features)
        model = train_cnn(model, training_imgs_for_cnn, training_features_for_cnn)
        
        print("********** Batch n°{} completed **********".format(i+1))
        
    return model

def train(number_of_batch, batch_size, name = 'im2gps_cnn'):
    if utils.check_cnn(name):
        model = utils.load_cnn(name)
    else:
        model = get_cnn_structure()
        utils.write_cnn(model, name)
        
    train_features_for_knn, train_gps_for_knn, train_ids_for_knn = data_import.get_training_data_for_knn()
    model = cnn_learning(model, number_of_batch, batch_size, train_ids_for_knn, train_features_for_knn)
    utils.delete_cnn(name)
    utils.write_cnn(model, name)
    return train_features_for_knn, train_gps_for_knn, train_ids_for_knn

def get_empty_cnn(name = 'im2gps_cnn_empty'):
    if utils.check_cnn(name):
        model = utils.load_cnn(name)
    else:
        model = get_cnn_structure()
        utils.write_cnn(model, name)
        
def apply(model, data):
    
    batch_size = 10
    n = len(data)
    i = 0
    
    output = []
    
    while (batch_size * i) < n:
        
        min_border = batch_size * i
        max_border = batch_size * (i + 1)
        
        if max_border >= n:
            max_border = n
            
        data_to_apply = data[min_border : max_border, :, :, :]
        output_data = model(data_to_apply)
        output.append(output_data)
        
        i += 1
    
    output = np.concatenate(output, axis = 0)
        
    return output
    