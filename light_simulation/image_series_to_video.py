import os
import glob
import imageio.v3 as iio

def create_gif_from_folder(image_folder, output_gif, fps=24):
    """
    Finds all matching spiral images in the folder, sorts them, and creates a GIF.
    
    :param image_folder: Directory where images are stored
    :param output_gif: Path/filename for the output GIF
    :param fps: Frames per second for the animation
    """
    # Look for any file matching the pattern 'render_spiral_*.png' inside the folder
    search_pattern = os.path.join(image_folder, "render_spiral_*.png")
    
    # Grab all matching file paths and sort them alphabetically/numerically
    found_files = sorted(glob.glob(search_pattern))
    
    if not found_files:
        print(f"No matching images found in '{image_folder}' using pattern 'render_spiral_*.png'")
        return

    print(f"Found {len(found_files)} frames. Reading images...")
    
    images = []
    for file_path in found_files:
        try:
            frame = iio.imread(file_path)
            images.append(frame)
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Skipping. Error: {e}")

    if not images:
        print("No valid images could be read. GIF generation aborted.")
        return

    print(f"Compiling {len(images)} frames into {output_gif}...")
    
    # Calculate duration per frame in milliseconds
    frame_duration = 1000 / fps 
    
    # Write the GIF (loop=0 means infinite loop)
    iio.imwrite(output_gif, images, duration=frame_duration, loop=0)
    print(f"Success! GIF saved as '{output_gif}'")

# --- Configuration ---
FOLDER = "C:/work/slow_lamp/render_test/largest_animation/"
OUTPUT = "C:/work/slow_lamp/render_test/largest_animation/spiral_animation.gif"
FPS = 20  # Adjust speed as needed

if __name__ == "__main__":
    create_gif_from_folder(FOLDER, OUTPUT, fps=FPS)