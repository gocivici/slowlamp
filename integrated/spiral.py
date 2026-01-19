import numpy as np
import cv2
import datetime

def to_cv2_color(color):
    """Converts (R, G, B) to OpenCV (B, G, R) format."""
    return (int(color[2]), int(color[1]), int(color[0]))

def drawSpiral(spiral_data, img_size=240, min_radius=40, max_radius=120):


    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    circle_center = (img_size // 2, img_size // 2)
    angle_per_hour = 360 / 24


    spiral_data.sort(key=lambda x: x['timestamp'])
    
    start_ts = spiral_data[0]['timestamp']
    end_ts = spiral_data[-1]['timestamp']
    duration_seconds = max(end_ts - start_ts, 1)

    # Calculate growth factor 'b'
    total_hours_duration = duration_seconds / 3600
    max_cumulative_angle_rad = (total_hours_duration * angle_per_hour) * (np.pi / 180)
    

    if max_cumulative_angle_rad < 2 * np.pi:
        max_cumulative_angle_rad = 2 * np.pi

    b = (max_radius - min_radius) / max_cumulative_angle_rad
    thickness = b * 2 * np.pi


    for i in range(len(spiral_data) - 1, -1, -1):
        entry = spiral_data[i]
        

        trace_ambient = to_cv2_color(entry.get('ac1', (128, 128, 128)))
        trace_vibrant = to_cv2_color(entry.get('vc', (255, 255, 255)))
        
 
        current_ts = entry['timestamp']
        hours_elapsed = (current_ts - start_ts) / 3600.0
        cumulative_angle_rad = (hours_elapsed * angle_per_hour) * (np.pi / 180)
        radius = int(min_radius + b * cumulative_angle_rad)
        

        dt_obj = datetime.datetime.fromtimestamp(current_ts)
        draw_angle_deg = (dt_obj.hour * angle_per_hour) - 90
        
        cv2.ellipse(
            canvas, circle_center, (radius, radius), 
            draw_angle_deg, 0, angle_per_hour, trace_ambient, -1
        )
        

        vibrant_radius = int(radius * 0.5) if radius < thickness else int(radius - (thickness * 0.5))
        cv2.ellipse(
            canvas, circle_center, (vibrant_radius, vibrant_radius), 
            draw_angle_deg, 0, angle_per_hour, trace_vibrant, -1
        )

    last_entry = spiral_data[-1]
    center_ac = to_cv2_color(last_entry.get('ac1', (128, 128, 128)))
    center_vc = to_cv2_color(last_entry.get('vc', (255, 255, 255)))

    cv2.circle(canvas, circle_center, 20, center_ac, -1)
    cv2.circle(canvas, circle_center, int(20 * 0.5), center_vc, -1)

    return canvas