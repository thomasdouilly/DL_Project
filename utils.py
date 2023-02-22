import os
import keras
import shutil
import numpy as np
from math import sin, cos, acos
from scipy.ndimage import gaussian_filter
#import reverse_geocoder
import folium
import webbrowser

def check_cnn(name = 'im2gps_cnn'):
    files = os.listdir('.')
    if name in files:
        cnn_check = True
    else:
        cnn_check = False
    return cnn_check

def delete_cnn(name = 'im2gps_cnn'):
    shutil.rmtree('./' + name, ignore_errors=True)
    
def write_cnn(model, name = 'im2gps_cnn'):
    model.save(name)
    
def load_cnn(name = 'im2gps_cnn'):
    model = keras.models.load_model('./' + name)
    return model

def normalize(data):
    data = np.array(data)
    N_output = data.shape[0]

    for n in range(N_output):
        feature = data[n, :]
        norm = np.linalg.norm(feature)
        if norm > 0:
            data[n, :] = feature / norm
    return data

def reshape(data):
    list_x = []
    for x in data[:2500000]:
      list_x.append(np.reshape(x, (x.shape[0], 1)))

    data = np.concatenate(list_x, axis = 1).T
    return data

def dist(point_1, point_2):
    (latitude_1, longitude_1) = point_1
    (latitude_2, longitude_2) = point_2

    latitude_1 *= np.pi/180
    latitude_2 *= np.pi/180
    longitude_1 *= np.pi/180
    longitude_2 *= np.pi/180

    delta_long = longitude_2 - longitude_1

    cos_S = sin(latitude_1) * sin(latitude_2) + cos(latitude_1) * cos(latitude_2) * cos(delta_long)
    S = acos(cos_S)

    distance = 6378.137 * S

    return distance

def calculations(output, found_dist, found_gps):
    query_outputs = {}

    for i in range(output.shape[0]):

        print("Loading : ", 100 * (i + 1) / output.shape[0], " %")
        actual_distance = found_dist[i]
        est_gps = found_gps[i]
        n = est_gps
        
        try:
            est_gps = np.reshape(est_gps, (n, 2))
        except:
            0
        
        w = np.zeros((180, 360))

        for j in range(750):
            latitude =  np.ceil(est_gps[j][0]) + 90 - 1
            longitude = np.ceil(est_gps[j][1]) + 180 - 1

            latitude = int(latitude)
            longitude = int(longitude)

            if actual_distance[j] != 0:
                w[latitude, longitude] += 1 / (actual_distance[j])**10

            w = gaussian_filter(w, 4)
    
        (current_latitude_x, current_longitude_x) = np.nonzero(np.where(w == w.max(), 1, 0))
        current_latitude_x = int(current_latitude_x)
        current_longitude_x = int(current_longitude_x)
        
        query_outputs[i] = [current_latitude_x - 90 + 1, current_longitude_x - 180 + 1]
        
        current_latitude_x = max(21, min(159, current_latitude_x))
        current_longitude_x = max(21, min(339, current_longitude_x))
        
        w = np.zeros((1800, 3600))

        for j in range(750):
            latitude =  np.ceil(est_gps[j][0] * 10) + 900 - 1
            longitude = np.ceil(est_gps[j][1] * 10) + 1800 - 1

            latitude = int(latitude)
            longitude = int(longitude)
            
            if actual_distance[j] != 0:
                w[latitude, longitude] += 1 / (actual_distance[j])**10


        w2 = gaussian_filter(w[current_latitude_x * 10 - 200 : current_latitude_x * 10 + 201, current_longitude_x * 10 - 200 : current_longitude_x * 10 + 201], 40)
        w = np.zeros((1800, 3600))
        w[current_latitude_x * 10 - 200 : current_latitude_x * 10 + 201, current_longitude_x * 10 - 200 : current_longitude_x * 10 + 201] = w2
    
        
        (latitude, longitude) = np.nonzero(np.where(w == w.max(), 1, 0))
        query_outputs[i] = np.array([latitude[0] - 900 + 1, longitude[0] - 1800 + 1]) /10
        
        d_best = np.inf
        j_best = - np.inf
        for j in range(750):
            gps1 = query_outputs[i]
            gps2 = est_gps[j]
            d = dist((gps1[0], gps1[1]), (gps2[0], gps2[1]))
            if d < 100:
                if d < d_best:
                    d_best = d
                    j_best = j
                    break

        if j_best >= 0:
            query_outputs[i] = est_gps[j_best]
            
    return query_outputs
    
def visualize_results(files, query_outputs):
    map_osm = folium.Map(location=[48.85, 2.34])
    for x in range(len(files)):
        print("The picture : ", files[x], " was geolocated : ", query_outputs[x])#, "(", reverse_geocoder.search(tuple(query_outputs[x]))[0]['cc'], ")")
        map_osm.add_child(folium.RegularPolygonMarker(location=[query_outputs[x][0], query_outputs[x][1]], fill_color='#132b5e', radius=5))
    map_osm.save("map_saved.html")
    webbrowser.open("map_saved.html")