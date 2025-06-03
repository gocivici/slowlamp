from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


data = np.load("traces/pairings-10-first-half.npz")
input_rgbs = data["pixel_colors"]
observed_rgbs = data["result_colors"]

pixel_colors = []
for r in range(0, 130, 10):
    for g in range(0, 255, 10):
        for b in range(0, 255, 10):
            pixel_colors.append( (r, g, b))

X_length = len(pixel_colors)

X = np.array(pixel_colors) / 255.0
y = np.array(observed_rgbs[:X_length]) / 255.0
model = LinearRegression().fit(X, y)

# To correct future colors:
def corrected_rgb(input_rgb):
    rgb_norm = np.array(input_rgb).reshape(1, -1) / 255.0
    corrected = model.predict(rgb_norm).clip(0, 1) * 255
    return corrected.astype(int)

r_pos = range(0, 130, 10)
g_pos = range(0, 255, 10)
b_pos = range(0, 255, 10)
test_id = 0

i_size = 10
j_size = 6
num_types = 2
interweave = False

for r in r_pos:
    canvas = np.ones((len(g_pos)*i_size, len(b_pos)*j_size*num_types, 3)).astype(np.uint8)
    for gi, g in enumerate(g_pos):
        for bi, b in enumerate(b_pos):
            pixel_color = (r, g, b)
            observed_color = y[test_id] * 255
            test_id+=1
            corrected_color = corrected_rgb(pixel_color)
            print(observed_color)
            if interweave:
                canvas[gi*i_size:(gi+1)*i_size, bi*j_size*num_types: bi*j_size*num_types+j_size ] = pixel_color
                canvas[gi*i_size:(gi+1)*i_size, bi*j_size*num_types+j_size: bi*j_size*num_types+j_size*2 ] = observed_color
            else:
                canvas[gi*i_size:(gi+1)*i_size, bi*j_size: bi*j_size+j_size ] = pixel_color
                canvas[gi*i_size:(gi+1)*i_size, len(b_pos)*j_size+bi*j_size: len(b_pos)*j_size+bi*j_size+j_size ] = observed_color


            # canvas[gi*i_size:(gi+1)*i_size, bi*j_size*num_types+j_size*2: bi*j_size*num_types+j_size*3 ] = corrected_color
            
    plt.imshow(canvas)
    plt.show()        
    
    