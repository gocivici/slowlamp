from colormath.color_objects import sRGBColor, XYZColor, xyYColor, LabColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



data = np.load("traces/pairings-32.npz")
input_rgbs = data["pixel_colors"].astype(np.int32)
observed_rgbs = data["result_colors"]

def rgb_to_lab(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, is_upscaled=False)
    lab = convert_color(srgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)

def rgb_to_xy(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, is_upscaled=False)
    xyz = convert_color(srgb, XYZColor)
    xyy = convert_color(xyz, xyYColor)
    # test = xyYColor()
    return (xyy.xyy_x, xyy.xyy_y)

def rgb_to_xyY(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, is_upscaled=False)
    xyz = convert_color(srgb, XYZColor)
    xyy = convert_color(xyz, xyYColor)
    # test = xyYColor()
    return (xyy.xyy_x, xyy.xyy_y, xyy.xyy_Y)

input_rgbs_list = input_rgbs.tolist()
observed_rgbs_list = observed_rgbs.tolist()
# Your NeoPixel RGB samples → xy coords
neopixel_xy = [rgb_to_xy(rgb) for rgb in observed_rgbs_list]
x_vals, y_vals = zip(*neopixel_xy)

# sRGB primaries and white point (for triangle)
sRGB_triangle = [
    (0.640, 0.330),  # Red
    (0.300, 0.600),  # Green
    (0.150, 0.060),  # Blue
    (0.640, 0.330)   # Close the triangle
]
triangle_x, triangle_y = zip(*sRGB_triangle)

plt.figure(figsize=(6, 6))
plt.plot(triangle_x, triangle_y, 'k-', label='sRGB Gamut')
# plt.scatter(x_vals, y_vals, c='red', alpha= 0.3, s=15, label='NeoPixel Captures')
plt.scatter(x_vals, y_vals, c=observed_rgbs/255, alpha= 0.6, s=15, label='NeoPixel Captures')
plt.xlabel('x')
plt.ylabel('y')
plt.title('NeoPixel vs sRGB Gamut in CIE 1931 xy')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# Convert all to Lab
# lab_vals = [rgb_to_lab(rgb) for rgb in observed_rgbs_list]
# L_vals, a_vals, b_vals = zip(*lab_vals)


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(L_vals, a_vals, b_vals, c='blue', s=5)
# ax.set_xlabel('L*')
# ax.set_ylabel('a*')
# ax.set_zlabel('b*')
# ax.set_title('NeoPixel Color Gamut in CIELAB')
# plt.show()

## shifts 

input_labs = [rgb_to_lab(rgb) for rgb in input_rgbs_list]
obs_labs = [rgb_to_lab(rgb) for rgb in observed_rgbs_list]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot input and observed points
for inp, obs in zip(input_labs, obs_labs):
    # Arrow from input to observed
    ax.quiver(inp[0], inp[1], inp[2],
              obs[0]-inp[0], obs[1]-inp[1], obs[2]-inp[2],
              arrow_length_ratio=0.1, color='gray', linewidth=1)

# Also scatter the points for clarity
L_in, a_in, b_in = zip(*input_labs)
L_obs, a_obs, b_obs = zip(*obs_labs)

ax.scatter(L_in, a_in, b_in, c='blue', label='Intended (Input)', s=30)
ax.scatter(L_obs, a_obs, b_obs, c='red', label='Observed (Camera)', s=30)

ax.set_xlabel('L*')
ax.set_ylabel('a*')
ax.set_zlabel('b*')
ax.set_title('Color Shifts: NeoPixel Input vs Camera Observed (CIELAB)')
ax.legend()
plt.show()


input_xyY = [rgb_to_xyY(rgb) for rgb in input_rgbs_list]
obs_xyY = [rgb_to_xyY(rgb) for rgb in observed_rgbs_list]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot input and observed points
for inp, obs in zip(input_xyY, obs_xyY):
    # Arrow from input to observed
    ax.quiver(inp[0], inp[1], inp[2],
              obs[0]-inp[0], obs[1]-inp[1], obs[2]-inp[2],
              arrow_length_ratio=0.1, color='gray', linewidth=1)

# Also scatter the points for clarity
x_in, y_in, Y_in = zip(*input_xyY)
x_obs, y_obs, Y_obs = zip(*obs_xyY)

# ax.scatter(x_in, y_in, Y_in, c='blue', label='Intended (Input)', s=30)
ax.scatter(x_in, y_in, Y_in, c=input_rgbs/255, label='Intended (Input)', s=30)
# ax.scatter(x_obs, y_obs, Y_obs, c='red', label='Observed (Camera)', s=30)
ax.scatter(x_obs, y_obs, Y_obs, c=observed_rgbs/255, label='Observed (Camera)', s=30)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Y')
ax.set_title('Color Shifts: NeoPixel Input vs Camera Observed (CIE 1931 xyY)')
ax.legend()
plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(122)
ax.plot(triangle_x, triangle_y, 'k-', label='sRGB Gamut')
# plt.scatter(x_vals, y_vals, c='red', alpha= 0.3, s=15, label='NeoPixel Captures')
ax.scatter(x_vals, y_vals, c=observed_rgbs/255, alpha= 0.6, s=15, label='NeoPixel Captures')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('NeoPixel output in CIE 1931 xy')
ax.grid(True)


x_vals, y_vals, _ = zip(*input_xyY)
ax = fig.add_subplot(121)
ax.plot(triangle_x, triangle_y, 'k-', label='sRGB Gamut')
# plt.scatter(x_vals, y_vals, c='red', alpha= 0.3, s=15, label='NeoPixel Captures')
ax.scatter(x_vals, y_vals, c=input_rgbs/255, alpha= 0.6, s=15, label='NeoPixel Captures')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('input sRGB in CIE 1931 xy')
ax.grid(True)

# plt.axis('equal')
plt.show()