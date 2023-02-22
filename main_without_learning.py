import knn
import cnn
import utils
import data_import
import time
import numpy as np

model = cnn.get_cnn_structure()

names, img = data_import.get_testing_data(200)

output = model(img)
output = utils.normalize(output)

_, train_gps_for_knn, train_ids_for_knn = data_import.get_training_data_for_knn()
training_imgs_for_knn, training_gps_for_knn = data_import.get_training_pictures_for_cnn(5000, train_ids_for_knn, train_gps_for_knn)

start_time = time.time()
training_features_for_knn = cnn.apply(model, training_imgs_for_knn)

found_dist, found_gps, found_features = knn.knn(training_features_for_knn, output, training_gps_for_knn, training_features_for_knn)
final_output = utils.calculations(output, found_dist, found_gps)
end_time = time.time()

print('***** Computations completed in {} seconds *****'.format(np.ceil(end_time - start_time)))


utils.visualize_results(names, final_output)