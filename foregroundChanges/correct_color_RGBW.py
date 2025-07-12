from colormath.color_objects import sRGBColor, XYZColor, xyYColor, LabColor
from colormath.color_conversions import convert_color
# import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from scipy.spatial import KDTree
# from scipy.optimize import minimize

import pickle
import os


data = np.load("pairings-32-tr4w2.npz")
input_rgbs = data["pixel_colors"].astype(np.int32)
observed_rgbs = data["result_colors"]

def normalize_lab(lab):
    l, a, b = lab
    return (l/100, (a+128)/255, (b+128)/255)

def rgb_to_lab_normalized(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, is_upscaled=False)
    lab = convert_color(srgb, LabColor)
    return normalize_lab((lab.lab_l, lab.lab_a, lab.lab_b))

def rgb_to_lab(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, is_upscaled=False)
    lab = convert_color(srgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)

input_rgbs_list = input_rgbs.tolist()
observed_rgbs_list = observed_rgbs.tolist()

X_rgbw = input_rgbs/255
# Your NeoPixel RGB samples → xy coords
Y_lab = [rgb_to_lab_normalized(rgb) for rgb in observed_rgbs_list]

observed_lab_data = np.array([rgb_to_lab(rgb) for rgb in observed_rgbs_list])
tree = KDTree(observed_lab_data)


inverse_model = None

if not os.path.exists('model.pkl'):
    ## machine learning
    inverse_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000)
    inverse_model.fit(Y_lab, X_rgbw)  # Predict RGBW from target Lab

    # save
    with open('model.pkl','wb') as f:
        pickle.dump(inverse_model,f)
else:
    with open('model.pkl', 'rb') as f:
        inverse_model = pickle.load(f)


def find_distance(target_rgba):
    query_point = rgb_to_lab(target_rgba[:3])
    distance, index = tree.query(query_point)
    print(f"Nearest neighbor is at index {index} with distance {distance}")
    return distance
    

## random forest and minimize does not work, for some reason... 
# forward_model = RandomForestRegressor()
# forward_model.fit(X_rgbw, Y_lab)  # Predict observed Lab from RGBW input
# does not work, for some reason... 
# def invert_lab_to_rgbw(target_lab):
#     def objective(rgbw):  # rgbw is a 4D vector in [0,1]
#         pred_lab = forward_model.predict([rgbw])[0]
#         return np.linalg.norm(np.array(pred_lab) - np.array(target_lab))  # or use CIEDE2000

#     result = minimize(objective, x0=[0.5, 0.5, 0.5, 0.0], bounds=[(0, 1)]*4) #RGBW 
#     # result = minimize(objective, x0=[0.5, 0.5, 0.5], bounds=[(0, 1)]*3)
#     return result.x  # best RGBW


def correct_color(target_rgba):
    global inverse_model

    query_point = rgb_to_lab_normalized(target_rgba[:3])
    pixel_input = inverse_model.predict([query_point])[0]
    pixel_input = (np.clip(pixel_input, 0, 1)*255).round()
    return (int(pixel_input[0]), int(pixel_input[1]) , int(pixel_input[2]), int(pixel_input[3]))

if __name__ == "__main__":
    # desired_lab = normalize_lab((70, 30, 15))
    desired_lab = Y_lab[100]
    # predicted_rgbw_forest = invert_lab_to_rgbw(desired_lab)
    predicted_rgbw_ml = inverse_model.predict([desired_lab])[0]

    print(X_rgbw[100])
    # print(predicted_rgbw_forest)
    print(predicted_rgbw_ml)

    ml_error = 0
    # forest_error = 0

    # exit()

    for i in range(len(Y_lab[::10])):
        desired_lab = Y_lab[i*10]
        # predicted_rgbw_forest = invert_lab_to_rgbw(desired_lab)
        predicted_rgbw_ml = inverse_model.predict([desired_lab])[0]
        print("observed", X_rgbw[i*10])
        # print("random f", predicted_rgbw_forest)
        print("mlp", predicted_rgbw_ml)
        ml_error += np.sum(np.abs(predicted_rgbw_ml-X_rgbw[i*10]))
        # forest_error += np.sum(np.abs(predicted_rgbw_forest-X_rgbw[i*10]))

    print("average ml error", ml_error/i/3)                                             
# print("average forest error", forest_error/i/3)
