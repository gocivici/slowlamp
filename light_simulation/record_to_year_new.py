import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
from datetime import datetime, timedelta
import cover_sim
from zoneinfo import ZoneInfo

# record = "C:/work/slow_lamp/light_simulation/20250723_five_days.txt"
record_files = ["20251110_235248_fgc_yumeng.txt", "20260223_082940_fgc_integrated.txt", 
                "20260317_084850_fgc_integrated.txt", "20260401_111140_fgc_integrated.txt",
                "20260402_083907_fgc_integrated.txt"]

# storage_file = open("C:/work/slow_lamp/light_simulation/simulate_year_with_half.txt", "w") 

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

class Trace:
    def __init__(self, main_color, day, hour, count = 5, traces_storing_mode="single", supplemental_colors=None, supplemental_counts=None):
        self.main_color = np.array(main_color[:3])
        self.main_count = count
        self.traces_storing_mode = traces_storing_mode
        self.supplemental_colors = []
        if traces_storing_mode != "single":
            self.supplemental_colors = np.array(supplemental_colors)[:, :3]
            self.supplemental_counts = supplemental_counts
        # self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        start_date = datetime.datetime(2025, 1, 1, 0, 0, 0 )
        dt = start_date + timedelta(days=day, hours=hour)
        self.timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")

    def print_trace(self):
        main_str = f'mode: {self.traces_storing_mode}; \
main_color (rgb) #{self.main_count}: ({int(self.main_color[0])}, {int(self.main_color[1])}, {int(self.main_color[2])});'
        supp_str = ''
        if self.traces_storing_mode != "single":
            supp_str = "supplemental_colors: ["
            for color, count in zip(self.supplemental_colors, self.supplemental_counts):
                supp_str += f' #{count} ({int(color[0])}, {int(color[1])}, {int(color[2])}),'
            supp_str+="];"
        return f'{main_str} {supp_str} @ {self.timestamp} \n'
    
    def paint_trace(self):
        global color_swatch_size
        swatch = np.ones((color_swatch_size, color_swatch_size*(1+len(self.supplemental_colors)), 3)).astype(np.uint8)
        swatch[:, :color_swatch_size] = self.main_color[::-1]
        if self.traces_storing_mode != "single":
            for i, color in enumerate(self.supplemental_colors):
                swatch[:, color_swatch_size*(i+1): color_swatch_size*(i+2)] = color[::-1]

        return swatch
    
sunrise_vibrant_colors = []
day_vibrant_colors = []
sunset_vibrant_colors = []
night_vibrant_colors = []

sunrise_ambient_colors = []
day_ambient_colors = []
sunset_ambient_colors = []
night_ambient_colors = []

sunrise_times = [8, 7.5, 7, 6.2, 5.7, 5, 5.5, 6.1, 6.8, 7.5, 7.5, 8]
sunset_times = [16.6, 17.5, 18.8, 20.2, 20.6, 21.2, 20.8, 20.6, 19.4, 18.4, 17, 16.3]
data = {}
start_hour = None

for j, filename in enumerate(record_files):
    record = "./light_simulation/"+filename
    with open(record, 'r') as file:
        for line in file:
            # mode: vaooo; main_color (rgb) #18779: (102, 128, 172); supplemental_colors: [ #51700 (21, 22, 20), #29686 (40, 41, 42), #15506 (71, 85, 115), #5918 (189, 213, 228),]; @ 2025-07-23 16:42:07 
            main_match = re.search(r"main_color \(rgb\) #(\d+): \((\d+), (\d+), (\d+)\)", line)
            if main_match:
                main_count = int(main_match.group(1))
                main_rgb = (int(main_match.group(2)), int(main_match.group(3)), int(main_match.group(4)))
                if main_rgb[0] == 255 and main_rgb[1] == 255 and main_rgb[2] == 255:
                    continue
                main_rgb_w_count = (int(main_match.group(1)), int(main_match.group(2)), int(main_match.group(3)), int(main_match.group(4)))
            # Extract all colors using regex
            matches = re.findall(r"#(\d+)\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
            if matches:
                colors_w_count = [ (int(count), int(r), int(g), int(b)) for count, r, g, b in matches ]
                print(colors_w_count)
                supp_colors_w_count = colors_w_count

            # time_match = re.search(r"@ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            # if time_match:
            #     time_str = time_match.group(1)
            #     dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            #     print("at hour", time_str)
            #     if start_hour is None:
            #         start_hour = dt
            #     hour = (dt-start_hour).total_seconds()//(60*60)

            time_match = re.search(r"@ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if time_match:
                time_str = time_match.group(1)
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                decimal_hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)
                month = dt.month - 1
                if decimal_hour >= sunrise_times[month]-1 and decimal_hour <= sunrise_times[month]+1:
                    sunrise_ambient_colors.append(supp_colors_w_count)
                    sunrise_vibrant_colors.append(main_rgb_w_count)
                elif decimal_hour >= sunrise_times[month]+1 and decimal_hour <= sunset_times[month]-1:
                    day_ambient_colors.append(supp_colors_w_count)
                    day_vibrant_colors.append(main_rgb_w_count)
                elif decimal_hour >= sunset_times[month]-1 and decimal_hour <= sunset_times[month]+1:
                    sunset_ambient_colors.append(supp_colors_w_count)
                    sunset_vibrant_colors.append(main_rgb_w_count)
                else:
                    night_ambient_colors.append(supp_colors_w_count)
                    night_vibrant_colors.append(main_rgb_w_count)

                if start_hour is None:
                    start_hour = dt

                hour_from_start = (dt-start_hour).total_seconds()//(60*60)
                record = [main_rgb_w_count[1:], main_rgb_w_count[0],
                        supp_colors_w_count[0][1:], supp_colors_w_count[0][0], supp_colors_w_count[1][1:], supp_colors_w_count[1][0],
                        supp_colors_w_count[2][1:], supp_colors_w_count[2][0], supp_colors_w_count[3][1:], supp_colors_w_count[3][0], hour_from_start]
                data[hour_from_start] = record

print("sunrise_ambient_colors", sunrise_ambient_colors)

start_day = 0
end_day = 365
start_dt = datetime(2025, 11, 11, 0, 0, 0)
# anchors = [0, 60, 70, 120, 180, 240, 300, 360]
# sunrise = [7, 7, 6, 6, 4, 6, 6, 7]
# sunrise_hours = np.interp(np.arange(0, 365), anchors, sunrise)
# dark_hours = np.random.randint(-1, 2, size=(365,))+22 

for i in range(start_day, end_day):
    # swatch = trace.paint_trace()
    current_date = start_dt + timedelta(days = i)

    sunrise_hour = sunrise_times[current_date.month-1]
    sunset_hour = sunset_times[current_date.month-1]
    
    for hour_in_day in range(0, 24):

        current_hour = current_date + timedelta(hours = hour_in_day)
        hour_from_start = (current_hour-start_hour).total_seconds()//(60*60)

        if hour_from_start%(365*24) in data:
            data_entry = data[hour_from_start][:-1]
            # cover_sim.save(*data_entry, hour_from_start)
            print("data exists", current_hour.month, current_hour.day, hour_in_day)
        else:
            mirro_axis = datetime(current_hour.year+1, 12, 21, 0, 0, 0)
            mirror_hour = ((mirro_axis - current_hour).total_seconds()//(60*60))%(365*24)
            adj_hour = 23 - mirror_hour%24
            mirror_hour = mirror_hour - mirror_hour%24 + adj_hour
            if mirror_hour in data:
                data_entry = data[mirror_hour][:-1]
                # cover_sim.save(*data_entry, hour_from_start)
                print("mirror exists", current_hour.month, current_hour.day, hour_in_day)
            else:
                if abs(hour_in_day - sunrise_hour)<=1:
                    color_choice = np.random.randint(0, len(sunrise_ambient_colors))
                    am_color = sunrise_ambient_colors[color_choice]
                    vr_color = sunrise_vibrant_colors[color_choice]
                elif hour_in_day > sunrise_hour + 1 and hour_in_day < sunset_hour - 1: 
                    color_choice = np.random.randint(0, len(day_ambient_colors))
                    am_color = day_ambient_colors[color_choice]
                    vr_color = day_vibrant_colors[color_choice]
                elif abs(hour_in_day - sunset_hour) <= 1: 
                    color_choice = np.random.randint(0, len(sunset_ambient_colors))
                    am_color = sunset_ambient_colors[color_choice]
                    vr_color = sunset_vibrant_colors[color_choice]
                else:
                    color_choice = np.random.randint(0, len(night_ambient_colors))
                    am_color = night_ambient_colors[color_choice]
                    vr_color = night_vibrant_colors[color_choice]
        
                print("am_color", am_color, len(sunrise_ambient_colors))
                # trace = Trace(am_color, i, hour, traces_storing_mode="levc", supplemental_colors=[vr_color, vr_color, vr_color])
                supplemental_counts = [color[0] for color in am_color]
                supplemental_colors = [[color[1], color[2], color[3]] for color in am_color]

                data_entry = [vr_color[1:], vr_color[0],
                        supplemental_colors[0], supplemental_counts[0], supplemental_colors[1], supplemental_counts[1],
                        supplemental_colors[2], supplemental_counts[2], supplemental_colors[3], supplemental_counts[3]]
        
        current_hour_utc = current_hour.replace(tzinfo=ZoneInfo("America/Vancouver"))
        cover_sim.save(*data_entry, current_hour_utc.timestamp()//3600)

#         trace = Trace(vr_color[1:], i, hour, count = vr_color[0], traces_storing_mode="vaooo", 
#                     supplemental_colors=supplemental_colors,
#                     supplemental_counts = supplemental_counts)
#         storage_file.writelines([trace.print_trace()])

# storage_file.flush()
# storage_file.close()
# glowing_canvas = (glowing_canvas/np.max(glowing_canvas)*255).astype(np.uint8)
# cv2.imwrite("C:/work/slow_lamp/render_test/render_glowing_spiral.png", glowing_canvas)
