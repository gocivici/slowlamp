import numpy as np
import cv2
import datetime

def to_cv2_color(color):
    """Converts (R, G, B) to OpenCV (B, G, R) format."""
    return (int(color[2]), int(color[1]), int(color[0]))

def drawSpiral(spiral_data, img_size=240, hub_outer=20, hub_inner=10, max_radius=120, cycle_steps=24):
    """
    Standard drawing function.
    cycle_steps: 24 for hourly views (1 rotation = 1 day)
                 30 for all-time view (1 rotation = 1 month)
    """
    if not spiral_data:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    circle_center = (img_size // 2, img_size // 2)
    
    spiral_data.sort(key=lambda x: x['timestamp'])
    n_entries = len(spiral_data)

    # 1. ARC MATH
    # Each slice occupies a portion of the rotation based on the cycle
    angle_step = 360 / cycle_steps
    arc_sweep = angle_step 

    # 2. GROWTH MATH
    # Total angle the spiral will travel
    max_cumulative_angle_rad = (n_entries * angle_step) * (np.pi / 180)
    max_cumulative_angle_rad = max(max_cumulative_angle_rad, 2 * np.pi)

    # 3. THICKNESS MATH
    # We want more data to result in thinner rings to stay within bounds
    half_thickness = max(1, int((max_radius - hub_outer) / (n_entries / (cycle_steps/2) + 5)))
    draw_min = hub_outer + 2 * half_thickness
    b = (max_radius - draw_min) / max_cumulative_angle_rad

    for i in range(len(spiral_data) - 1, -1, -1):
        entry = spiral_data[i]
        trace_ambient = to_cv2_color(entry.get('ac1', (128, 128, 128)))
        trace_vibrant = to_cv2_color(entry.get('vc', (255, 255, 255)))

        # Distance from center
        cumulative_angle_rad = (i * angle_step) * (np.pi / 180)
        radius = int(draw_min + b * cumulative_angle_rad)

        # Position around the circle
        # Instead of timestamp % 24, we use the sequence index % cycle_steps
        # to ensure the spiral 'walks' correctly around the circle.
        start_angle = (i * angle_step) - 90
        end_angle = start_angle + arc_sweep

        inner_radius = max(1, radius - half_thickness)
        
        cv2.ellipse(canvas, circle_center, (radius, radius), 0, 
                    start_angle, end_angle, trace_ambient, -1)
        cv2.ellipse(canvas, circle_center, (inner_radius, inner_radius), 0, 
                    start_angle, end_angle, trace_vibrant, -1)

    # Hub (representing the most recent moment)
    last_entry = spiral_data[-1]
    cv2.circle(canvas, circle_center, hub_outer, to_cv2_color(last_entry.get('ac1')), -1)
    cv2.circle(canvas, circle_center, hub_inner, to_cv2_color(last_entry.get('vc')), -1)
    
    return canvas

def generate_slowlamp_views(all_data):
    all_data.sort(key=lambda x: x['timestamp'])
    latest_ts = all_data[-1]['timestamp']

    # VIEW 1: Last 24 Hours (Cycle of 24)
    view_24h = [d for d in all_data if d['timestamp'] > (latest_ts - 24)]
    
    # VIEW 2: Last 30 Days (Cycle of 24)
    view_30d = [d for d in all_data if d['timestamp'] > (latest_ts - (24 * 30))]

    # VIEW 3: All-Time (Daily Average, Cycle of 30)
    daily_bins = {}
    for entry in all_data:
        day_id = int(entry['timestamp'] // 24)
        if day_id not in daily_bins:
            daily_bins[day_id] = {'ac1': [], 'vc': []}
        daily_bins[day_id]['ac1'].append(entry['ac1'])
        daily_bins[day_id]['vc'].append(entry['vc'])

    view_all_time_avg = []
    for day_id in sorted(daily_bins.keys()):
        avg_ac = np.mean(daily_bins[day_id]['ac1'], axis=0).astype(int)
        avg_vc = np.mean(daily_bins[day_id]['vc'], axis=0).astype(int)
        
        view_all_time_avg.append({
            'timestamp': day_id * 24.0, 
            'ac1': tuple(avg_ac), 
            'vc': tuple(avg_vc)
        })
    
    # Draw them with different 'pulses'
    img_24h = drawSpiral(view_24h, cycle_steps=24)
    img_30d = drawSpiral(view_30d, cycle_steps=24)
    img_all_time = drawSpiral(view_all_time_avg, cycle_steps=30) # Monthly cycle

    return img_24h, img_30d, img_all_time