import bpy
import csv

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Add a camera
bpy.ops.object.camera_add(location=(0, 0, 30), rotation=(0, 0, 0))
camera = bpy.context.view_layer.objects.active  # ← safer than context.object
bpy.context.scene.camera = camera

## Add a point light
#bpy.ops.object.light_add(type='POINT', location=(0, 0, 2))
#lamp = bpy.context.view_layer.objects.active
#lamp.data.energy = 1000  # make it visible

## Add a point light
#bpy.ops.object.light_add(type='POINT', location=(1, 1, 2))
#lamp2 = bpy.context.view_layer.objects.active
#lamp2.data.energy = 500  # make it visible


# 1. Create lights
lights = []
with open("C:/work/slow_lamp/light_simulation/coordinates_freeform.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # skip header

    for row in reader:
        for i in range(0, len(row), 3):
            index = (i) // 3
            x = float(row[i])
            y = float(row[i + 1])
            z = float(row[i + 2])
            bpy.ops.object.light_add(type='POINT', location=(x, y, z))
            light = bpy.context.view_layer.objects.active
            light.name = f"Light_{index}"
            light.data.energy = 1000
            lights.append(light)
                
# 2. Create a name map for lookup
lights_by_name = {light.name: light for light in lights}

# 3. Load CSV file with colors per light per frame
def animate_from_csv(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header

        for row in reader:
            frame = int(float(row[0]))*15
            for i in range(1, len(row), 4):
                index = (i - 1) // 4
                if index >= len(lights):
                    continue
                watt = float(row[i])
                r = float(row[i + 1])
                g = float(row[i + 2])
                b = float(row[i + 3])
                light = lights[index]
                light.data.color = (r, g, b)
                # print(index, watt, r)
                light.data.keyframe_insert(data_path="color", frame=frame)
                light.data.energy = watt
                light.data.keyframe_insert(data_path="energy", frame=frame)
        return frame

# 🔁 Call the animation function with your CSV path
csv_path = "C:/work/slow_lamp/light_simulation/animation_plan_freeform.csv"  # ← change this
end_frame = animate_from_csv(csv_path)

# Add a plane (so we can see the light effect)
# bpy.ops.mesh.primitive_plane_add(size=11, location=(-0.5, -0.5, 0))

# Basic circular plane (filled circle)
bpy.ops.mesh.primitive_circle_add(
    radius= 6,                    # Half of your desired diameter (11/2)
    location=(0, 0, 0),
    fill_type='TRIFAN'           # Creates a filled circle
)

# Set frame rate and duration
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = end_frame + 15  
scene.render.fps = 15

## Insert keyframes for light color
#lamp.data.color = (1.0, 0.0, 0.0)  # red
#lamp.data.keyframe_insert(data_path="color", frame=1)

#lamp.data.color = (0.0, 0.0, 1.0)  # blue
#lamp.data.keyframe_insert(data_path="color", frame=125)

## Insert keyframes for light color
#lamp2.data.color = (1.0, 1.0, 0.0)  # red
#lamp2.data.keyframe_insert(data_path="color", frame=1)

#lamp2.data.color = (1.0, 0.0, 1.0)  # blue
#lamp2.data.keyframe_insert(data_path="color", frame=125)

## Optional: use smooth interpolation
#for fc in lamp.data.animation_data.action.fcurves:
#    for kp in fc.keyframe_points:
#        kp.interpolation = 'LINEAR'  # smoother color transition
#        
## Optional: use smooth interpolation
#for fc in lamp2.data.animation_data.action.fcurves:
#    for kp in fc.keyframe_points:
#        kp.interpolation = 'LINEAR'  # smoother color transition

# Use Eevee for real-time rendering
# scene.render.engine = 'BLENDER_EEVEE'
#scene.eevee.use_bloom = True  # Nice glow effect

# Set render settings
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.filepath = "//color_change_lamp_freeform.mp4"
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

# Trigger render (this will take some time)
# Uncomment the line below to render immediately
# bpy.ops.render.render(animation=True)
