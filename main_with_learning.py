import cnn
import knn
import utils
import data_import
import time
import numpy as np

#cnn.train(5, 128)

model = utils.load_cnn()
names, img = data_import.get_testing_data(50)

start_time = time.time()
output = model(img)
output = utils.normalize(output)

train_features_for_knn, train_gps_for_knn, train_ids_for_knn = data_import.get_training_data_for_knn()
train_features_for_knn = utils.reshape(train_features_for_knn)
train_features_for_knn = utils.normalize(train_features_for_knn)

found_dist, found_gps, found_features = knn.knn(train_features_for_knn, output, train_gps_for_knn, train_features_for_knn)

final_output = utils.calculations(output, found_dist, found_gps)

end_time = time.time()

print('***** Computations completed in {} seconds *****'.format(np.ceil(end_time - start_time)))

utils.visualize_results(names, final_output)