from colormath.color_objects import sRGBColor, XYZColor, xyYColor, LabColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from scipy.spatial import KDTree
# from scipy.optimize import minimize

import pickle
import os

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
    
def gamma_correct(i, gamma=2.4):
    i = np.clip( i, 0, 2**16-1)
    return np.round(((2**16-1)*(i/(2**16-1))**gamma))



inverse_model = None
model_name = 'hd108_joint_model_relu.pkl'
data_folder = './HD108/'
if not os.path.exists(model_name):
    data = np.load(data_folder+"pairings-32-black_HD108_777.npz")
    linear7_input_rgbs = data["pixel_colors"].astype(np.int32)*2**7
    linear7_observed_rgbs = data["result_colors"]

    data = np.load(data_folder+"pairings-32-black_HD108_875.npz")
    linear875_input_rgbs = data["pixel_colors"].astype(np.int32)
    linear875_input_rgbs[:, 0] = linear875_input_rgbs[:, 0]*2**8
    linear875_input_rgbs[:, 1] = linear875_input_rgbs[:, 1]*2**7
    linear875_input_rgbs[:, 2] = linear875_input_rgbs[:, 2]*2**5
    linear875_observed_rgbs = data["result_colors"]


    data = np.load(data_folder+"pairings-32-black_HD108_877-2.4.npz")
    gamma_input_rgbs = data["pixel_colors"].astype(np.int32)
    gamma_input_rgbs[:, 0] = gamma_input_rgbs[:, 0]*2**8
    gamma_input_rgbs[:, 1] = gamma_input_rgbs[:, 1]*2**7
    gamma_input_rgbs[:, 2] = gamma_input_rgbs[:, 2]*2**7
    gamma_input_rgbs = gamma_correct(gamma_input_rgbs)

    gamma_observed_rgbs = data["result_colors"]

    input_rgbs = np.concatenate([linear7_input_rgbs, linear875_input_rgbs, gamma_input_rgbs])
    observed_rgbs = np.concatenate([linear7_observed_rgbs, linear875_observed_rgbs, gamma_observed_rgbs])

    print("input_rgbs[11]", input_rgbs[11])
    input_rgbs_list = input_rgbs.tolist()
    observed_rgbs_list = observed_rgbs.tolist()

    X_rgbw = input_rgbs/2**16
    # Your NeoPixel RGB samples → xy coords
    Y_lab = [rgb_to_lab_normalized(rgb) for rgb in observed_rgbs_list]

    # observed_lab_data = np.array([rgb_to_lab(rgb) for rgb in observed_rgbs_list])
    # tree = KDTree(observed_lab_data)

    # visualize collected data
    if __name__ == "__main__":
        
        red_zero_mask = input_rgbs[:, 0] == 0
        green_zero_mask = input_rgbs[:, 1] == 0
        blue_zero_mask = input_rgbs[:, 2] == 0

        red_only_mask = green_zero_mask & blue_zero_mask
        green_only_mask = red_zero_mask & blue_zero_mask
        blue_only_mask = red_zero_mask & green_zero_mask

        red_only_inputs = input_rgbs[red_only_mask]/2**16
        green_only_inputs = input_rgbs[green_only_mask]/2**16
        blue_only_inputs = input_rgbs[blue_only_mask]/2**16

        red_only_outputs = observed_rgbs[red_only_mask]
        green_only_outputs = observed_rgbs[green_only_mask]
        blue_only_outputs = observed_rgbs[blue_only_mask]

        plt.figure(figsize=(9, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(red_only_inputs[:, 0], red_only_outputs[:, 0], color = "red")
        plt.scatter(red_only_inputs[:, 0], red_only_outputs[:, 1], color = "green")
        plt.scatter(red_only_inputs[:, 0], red_only_outputs[:, 2], color = "blue")
        plt.title("red 0 - 255")
        plt.xlim((-0.01, 1.01))
        plt.ylim([-1, 256])

        plt.subplot(1, 3, 2)
        plt.scatter(green_only_inputs[:, 1], green_only_outputs[:, 0], color = "red")
        plt.scatter(green_only_inputs[:, 1], green_only_outputs[:, 1], color = "green")
        plt.scatter(green_only_inputs[:, 1], green_only_outputs[:, 2], color = "blue")
        plt.title("green 0 - 255")
        plt.ylim([-1, 256])
        plt.xlim((-0.01, 1.01))

        plt.subplot(1, 3, 3)
        plt.scatter(blue_only_inputs[:, 2], blue_only_outputs[:, 0], color = "red")
        plt.scatter(blue_only_inputs[:, 2], blue_only_outputs[:, 1], color = "green")
        plt.scatter(blue_only_inputs[:, 2], blue_only_outputs[:, 2], color = "blue")
        plt.title("blue 0 - 255")
        plt.ylim([-1, 256])
        plt.xlim((-0.01, 1.01))

        plt.show()
    
    ## machine learning
    inverse_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000)
    inverse_model.out_activation_ = "relu"
    inverse_model.fit(Y_lab, X_rgbw)  # Predict RGBW from target Lab

    # save
    with open(model_name,'wb') as f:
        pickle.dump(inverse_model,f)
else:
    with open(model_name, 'rb') as f:
        inverse_model = pickle.load(f)
        inverse_model.out_activation_ = "relu"


# def find_distance(target_rgba):
#     query_point = rgb_to_lab(target_rgba[:3])
#     distance, index = tree.query(query_point)
#     print(f"Nearest neighbor is at index {index} with distance {distance}")
#     return distance
    

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
    inverse_model.out_activation_ = "relu"
    query_point = rgb_to_lab_normalized(target_rgba[:3])
    pixel_input = inverse_model.predict([query_point])[0]
    pixel_input = (np.clip(pixel_input, 0, 1)*(2**16-1)).round()
    return (int(pixel_input[0]), int(pixel_input[1]) , int(pixel_input[2]))

def correct_color_from_lab(lab_array):
    global inverse_model
    inverse_model.out_activation_ = "relu"
    lab_array[:, 0] /= 100 # (l/100, (a+128)/255, (b+128)/255)
    lab_array[:, 1] = (lab_array[:, 1]+128)/255 # (l/100, (a+128)/255, (b+128)/255)
    lab_array[:, 2] = (lab_array[:, 2]+128)/255  # (l/100, (a+128)/255, (b+128)/255)
    pixel_input = inverse_model.predict(lab_array)
    pixel_input = (np.clip(pixel_input, 0, 1)*(2**16-1)).round()
    return pixel_input

if __name__ == "__main__":
    # desired_lab = normalize_lab((70, 30, 15))
    desired_lab = Y_lab[100]
    # predicted_rgbw_forest = invert_lab_to_rgbw(desired_lab)
    inverse_model.out_activation_ = "relu"
    predicted_rgbw_ml = inverse_model.predict([desired_lab])[0]

    print("X_rgbw[100]", X_rgbw[100])
    # print(predicted_rgbw_forest)
    print("predicted_rgbw_ml", predicted_rgbw_ml)

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

    test = np.linspace(0, 255, 255)

    test_lab_colors = []
    for r in list(test):
        test_lab = rgb_to_lab_normalized((r, 0, 0))
        test_lab_colors.append(test_lab)

    predicted_r = inverse_model.predict(np.array(test_lab_colors))

    test_lab_colors = []
    for g in list(test):
        test_lab = rgb_to_lab_normalized((0, g, 0))
        test_lab_colors.append(test_lab)

    predicted_g = inverse_model.predict(np.array(test_lab_colors))

    test_lab_colors = []
    for b in list(test):
        test_lab = rgb_to_lab_normalized((0, 0, b))
        test_lab_colors.append(test_lab)

    predicted_b = inverse_model.predict(np.array(test_lab_colors))

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 3, 1)
    plt.plot(test, predicted_r[:, 0], color = "red")
    plt.plot(test, predicted_r[:, 1], color = "green")
    plt.plot(test, predicted_r[:, 2], color = "blue")
    plt.title("red 0 - 255")
    plt.ylim([-0.25, 1.1])

    plt.subplot(1, 3, 2)
    plt.plot(test, predicted_g[:, 0], color = "red")
    plt.plot(test, predicted_g[:, 1], color = "green")
    plt.plot(test, predicted_g[:, 2], color = "blue")
    plt.title("green 0 - 255")
    plt.ylim([-0.25, 1.1])

    plt.subplot(1, 3, 3)
    plt.plot(test, predicted_b[:, 0], color = "red")
    plt.plot(test, predicted_b[:, 1], color = "green")
    plt.plot(test, predicted_b[:, 2], color = "blue")
    plt.title("blue 0 - 255")
    plt.ylim([-0.25, 1.1])

    plt.show()
# print("average forest error", forest_error/i/3)
