import numpy as np
import cv2
import datetime

def to_cv2_color(color):
    """Converts (R, G, B) to OpenCV (B, G, R) format."""
    return (int(color[2]), int(color[1]), int(color[0]))

def drawSpiral(spiral_data, img_size=240, min_radius=40, max_radius=120, interval_minutes=60):

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    circle_center = (img_size // 2, img_size // 2)
    
    # --- CHANGED: Convert interval to hours for our math ---
    interval_hours = interval_minutes / 60.0 
    slices_per_day = 24 / interval_hours
    
    angle_per_slice = 360 / slices_per_day  # angular step between entries
    arc_sweep = 360 / 24  # fixed 15-degree arc width for visual clarity

    spiral_data.sort(key=lambda x: x['timestamp'])

    start_ts = spiral_data[0]['timestamp']
    end_ts = spiral_data[-1]['timestamp']
    
    # --- CHANGED: Calculate duration based on hour intervals ---
    duration_slices = max((end_ts - start_ts) / interval_hours, 1)

    # Calculate growth factor 'b'
    max_cumulative_angle_rad = (duration_slices * angle_per_slice) * (np.pi / 180)

    if max_cumulative_angle_rad < 2 * np.pi:
        max_cumulative_angle_rad = 2 * np.pi

    hub_outer = 20
    hub_inner = 10

    # half_thickness computed so that draw_min = hub_outer + 2*half_thickness fits in max_radius
    n = max(1, len(spiral_data))
    half_thickness = max(1, int((max_radius - hub_outer) * slices_per_day / (2 * n + 2 * slices_per_day)))
    draw_min = hub_outer + 2 * half_thickness
    b = (max_radius - draw_min) / max_cumulative_angle_rad

    for i in range(len(spiral_data) - 1, -1, -1):
        entry = spiral_data[i]

        trace_ambient = to_cv2_color(entry.get('ac1', (128, 128, 128)))
        trace_vibrant = to_cv2_color(entry.get('vc', (255, 255, 255)))

        current_ts = entry['timestamp']
        
        # --- CHANGED: Calculate elapsed slices in hours ---
        slices_elapsed = (current_ts - start_ts) / interval_hours
        cumulative_angle_rad = (slices_elapsed * angle_per_slice) * (np.pi / 180)
        radius = int(draw_min + b * cumulative_angle_rad)

        # --- CHANGED: current_ts is in hours, so mod 24 gives the hour of the day ---
        slice_of_day = (current_ts % 24) / interval_hours
        start_angle = (slice_of_day * angle_per_slice) - 90
        end_angle = start_angle + arc_sweep

        inner_radius = max(1, radius - half_thickness)

        # outer half of band: ambient (filled pie)
        cv2.ellipse(canvas, circle_center, (radius, radius),
                    0, start_angle, end_angle, trace_ambient, -1)
        # inner half of band: vibrant (filled pie, clips ambient)
        cv2.ellipse(canvas, circle_center, (inner_radius, inner_radius),
                    0, start_angle, end_angle, trace_vibrant, -1)

    last_entry = spiral_data[-1]
    center_ac = to_cv2_color(last_entry.get('ac1', (128, 128, 128)))
    center_vc = to_cv2_color(last_entry.get('vc', (255, 255, 255)))
    cv2.circle(canvas, circle_center, hub_outer, center_ac, -1)
    cv2.circle(canvas, circle_center, hub_inner, center_vc, -1)

    return canvas