from sklearn.linear_model import LinearRegression
import numpy as np

data = np.load("pairings.npz")
input_rgbs = data["pixel_colors"]
observed_rgbs = data["result_colors"]

X = np.array(input_rgbs) / 255.0
y = np.array(observed_rgbs) / 255.0
model = LinearRegression().fit(X, y)

# To correct future colors:
def corrected_rgb(input_rgb):
    rgb_norm = np.array(input_rgb).reshape(1, -1) / 255.0
    corrected = model.predict(rgb_norm).clip(0, 1) * 255
    return corrected.astype(int)