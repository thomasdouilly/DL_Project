from sklearn.neighbors import NearestNeighbors

def knn(training_data, input, gps_data, features_data): 
    tree = NearestNeighbors(n_neighbors = 750, n_jobs = 4)
    neighbors = tree.fit(training_data)
    
    distances, indices = neighbors.kneighbors(input)
    indices = indices.T
    distances = distances.T
    
    n_nearest = 1000
    found_dist = []
    found_gps = []
    found_features = []

    for i in range(input.shape[0]):
        print("Loading : ", 100 * (i + 1) / input.shape[0], " %")
        found_dist.append(distances[:n_nearest, i])
        found_gps.append(gps_data[indices[:n_nearest, i]])
        found_features.append(features_data[indices[:n_nearest, i]])
        
    return found_dist, found_gps, found_features