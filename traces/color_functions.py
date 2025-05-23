import numpy as np


def srgb_to_linear(rgb):
    rgb = np.array(rgb) / 255.0
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear_rgb

def linear_rgb_to_xyz(linear_rgb):
    # Transformation matrix for linear RGB to CIEXYZ
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    return np.dot(M, linear_rgb)

def xyz_to_lab(xyz):
    # Reference white point for D65 illuminant
    X_n, Y_n, Z_n = 95.047, 100.0, 108.883
    X, Y, Z = xyz / np.array([X_n, Y_n, Z_n])

    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), (t / (3 * delta**2)) + 4/29)

    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return np.array([L, a, b])

def srgb_to_lab(rgb):
    linear_rgb = srgb_to_linear(rgb)
    xyz = linear_rgb_to_xyz(linear_rgb)
    lab = xyz_to_lab(xyz)
    return lab

# Function to calculate CIEDE2000
def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    h1_prime += 360 if h1_prime < 0 else 0
    h2_prime += 360 if h2_prime < 0 else 0
    
    avg_H_prime = (h1_prime + h2_prime + 360) / 2.0 if np.abs(h1_prime - h2_prime) > 180 else (h1_prime + h2_prime) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(avg_H_prime - 30)) + 0.24 * np.cos(np.radians(2 * avg_H_prime)) + 0.32 * np.cos(np.radians(3 * avg_H_prime + 6)) - 0.20 * np.cos(np.radians(4 * avg_H_prime - 63))
    
    delta_h_prime = h2_prime - h1_prime
    delta_h_prime -= 360 if delta_h_prime > 180 else 0
    delta_h_prime += 360 if delta_h_prime < -180 else 0
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2.0)
    
    S_L = 1 + ((0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2))
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    
    R_T = -2 * np.sqrt(avg_C_prime**7 / (avg_C_prime**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-(((avg_H_prime - 275) / 25)**2))))
    
    delta_E = np.sqrt((delta_L_prime / S_L)**2 + (delta_C_prime / S_C)**2 + (delta_H_prime / S_H)**2 + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))
    return delta_E
    
if __name__ == "__main__": 
    # Example usage
    rgb1 = [255, 0, 0]  # Red
    lab1 = srgb_to_lab(rgb1)

    srgb2 = np.array([119, 119, 119])  # Gray
    linear_rgb2 = srgb_to_linear(srgb2)
    xyz2 = linear_rgb_to_xyz(linear_rgb2)
    lab2 = xyz_to_lab(xyz2)
    # lab2 = srgb_to_lab(rgb2)

    print(f"Lab1: {lab1}")
    print(f"RGB: {linear_rgb2}, xyz: {xyz2}, lab: {lab2}")



    # # Function to convert RGB to LAB
    # def rgb_to_lab(rgb):
    #     # Normalize the RGB values to [0, 1]
    #     rgb_normalized = np.array(rgb) / 255.0
    #     # Convert RGB to LAB using skimage's color module
    #     lab = color.rgb2lab(rgb_normalized[np.newaxis, np.newaxis, :])
    #     return lab[0, 0, :]

    

    # Example: Compare two RGB colors
    rgb1 = [255, 0, 0]   # Red
    rgb2 = [119, 119, 119]   # Green

    # Convert RGB to LAB
    lab1 = srgb_to_lab(rgb1)
    lab2 = srgb_to_lab(rgb2)

    # Calculate CIEDE2000 color difference
    delta_e = ciede2000(lab1, lab2)
    print(f"CIEDE2000 color difference: {delta_e:.4f}")


