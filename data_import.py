import numpy as np
import cv2
import os
from PIL import Image
import scipy.io
import flickrapi
import urllib
from PIL import Image
import time
import random as rd

   
def get_testing_data(number):
    print('***** Import of {} test pictures *****'.format(number))
    img = []

    files = os.listdir('./Photos_Test')
    np.random.shuffle(files)
    files = files[:number]

    for name in files:
        add_img = np.array(Image.open("./Photos_Test/" + name))
        add_img = cv2.resize(add_img, (224, 224), interpolation = cv2.INTER_AREA)
        add_img = np.expand_dims(add_img, axis = 0)
        img.append(add_img)

    img = np.concatenate(img, axis = 0).astype('float')
    print('***** Import completed *****')
    return files, img

def get_training_data_for_cnn(train_number, train_ids_for_knn, train_features_for_knn):
    
    print('***** Import of {} training pictures for CNN *****'.format(train_number))
    
    flickr = flickrapi.FlickrAPI('baa17527f437c547c79366be3f0eb6eb', 'bc26415188950b36', cache=True)

    N = len(train_ids_for_knn)

    train_imgs_for_cnn = []
    train_features_for_cnn = []

    start_time = time.time()
    for k in range(train_number):

        i = rd.randint(0, N-1)
        time.sleep(0.1)

        if 40*(k+1) // train_number == 40*(k+1) / train_number:
            print('Loading : ', 100 * (k+1) / train_number, ' %') 

        photo_ID = train_ids_for_knn[i].split('_')[2]

        try:
            photo = flickr.photos.getInfo(photo_id = photo_ID)[0]
        
        except:
            0
        
        else:
            farmId = photo.get("farm")
            serverId = photo.get("server")
            secretId = photo.get("secret")

            url = "https://farm" + farmId +".staticflickr.com/" + serverId + "/" + photo_ID + "_" + secretId +".jpg"

            with urllib.request.urlopen(url) as picture:
                img = Image.open(picture)
                img = np.array(img)

                if len(img.shape) == 2:
                    img = np.stack((img,)*3, axis=-1)
                    
                else:
                    if img.shape[-1] > 3:
                        img = img[:, :, :3]

                img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(img, axis = 0)

            train_features_for_cnn.append(train_features_for_knn[i].reshape((512,1)))
            train_imgs_for_cnn.append(img)

    end_time = time.time()
    
    training_imgs_for_cnn = np.concatenate(train_imgs_for_cnn, axis = 0).astype('float')
    training_features_for_cnn = np.concatenate(train_features_for_cnn, axis = 1).T

    print('***** Import completed in {} seconds *****'.format(np.ceil(end_time - start_time)))
    return training_imgs_for_cnn, training_features_for_cnn


def get_training_pictures_for_cnn(train_number, train_ids_for_knn, train_gps_for_knn):
    
    print('***** Import of {} training pictures for CNN *****'.format(train_number))
    
    flickr = flickrapi.FlickrAPI('baa17527f437c547c79366be3f0eb6eb', 'bc26415188950b36', cache=True)

    N = len(train_ids_for_knn)

    train_imgs_for_cnn = []
    train_gps_for_cnn = []

    start_time = time.time()
    for k in range(train_number):

        i = rd.randint(0, N-1)
        time.sleep(0.1)

        if 40*(k+1) // train_number == 40*(k+1) / train_number:
            print('Loading : ', 100 * (k+1) / train_number, ' %') 

        photo_ID = train_ids_for_knn[i].split('_')[2]

        try:
            photo = flickr.photos.getInfo(photo_id = photo_ID)[0]
        
        except:
            0
        
        else:
            farmId = photo.get("farm")
            serverId = photo.get("server")
            secretId = photo.get("secret")

            url = "https://farm" + farmId +".staticflickr.com/" + serverId + "/" + photo_ID + "_" + secretId +".jpg"

            with urllib.request.urlopen(url) as picture:
                img = Image.open(picture)
                img = np.array(img)

                if len(img.shape) == 2:
                    img = np.stack((img,)*3, axis=-1)
                    
                else:
                    if img.shape[-1] > 3:
                        img = img[:, :, :3]

                img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(img, axis = 0)
            print
            train_gps_for_cnn.append(train_gps_for_knn[i][0].reshape((2,1)))
            train_imgs_for_cnn.append(img)

    end_time = time.time()
    
    training_imgs_for_cnn = np.concatenate(train_imgs_for_cnn, axis = 0).astype('float')
    training_gps_for_cnn = np.concatenate(train_gps_for_cnn, axis = 1).T

    print('***** Import completed in {} seconds *****'.format(np.ceil(end_time - start_time)))
    return training_imgs_for_cnn, training_gps_for_cnn

def get_training_data_for_knn():
    
    print('***** Import of training dataset *****')
    
    files = os.listdir('./Photos_Train/Matrices')
    np.random.shuffle(files)
    files = files

    train_features = []
    train_gps = []
    train_ids = []

    for name in files:
        added_img = scipy.io.loadmat('./Photos_Train/Matrices/' + name)

        added_img_features = added_img['file_features'].flatten()
        added_img_gps = added_img['file_gps'].flatten()
        added_img_ids = added_img['file_ids'].flatten()

        train_features.append(added_img_features)
        train_gps.append(added_img_gps)
        train_ids.append(np.vectorize(lambda  x: x[0][61:].replace('/', '_'))(added_img_ids ))

    train_features_for_knn = np.concatenate(train_features, axis = 0)
    train_gps_for_knn = np.concatenate(train_gps, axis = 0)
    train_ids_for_knn = np.concatenate(train_ids, axis = 0)
    
    print('***** Import completed *****')
    
    return train_features_for_knn, train_gps_for_knn, train_ids_for_knn