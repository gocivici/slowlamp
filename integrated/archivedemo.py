import cover
import time
import random

NUM_CAPTURES = 24*7
base_timestamp = int(time.time())
HOUR = 3600

# DEBUG: fixed colors
# color_vibrant = [255, 0, 0]
# ambientColor1 = ambientColor2 = ambientColor3 = ambientColor4 = [0, 0, 255]
# count_vc = count_ac_1 = count_ac_2 = count_ac_3 = count_ac_4 = 500000

for i in range(NUM_CAPTURES):
    current_timestamp = base_timestamp + i * HOUR

    color_vibrant = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    ambientColor1 = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    ambientColor2 = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    ambientColor3 = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    ambientColor4 = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    count_vc = random.randint(0, 1000000)
    count_ac_1 = random.randint(0, 1000000)
    count_ac_2 = random.randint(0, 1000000)
    count_ac_3 = random.randint(0, 1000000)
    count_ac_4 = random.randint(0, 1000000)

    cover.save(
        color_vibrant, 0,
        ambientColor1, count_ac_1,
        ambientColor2, count_ac_2,
        ambientColor3, count_ac_3,
        ambientColor4, count_ac_4,
        current_timestamp
    )
    readable = time.strftime("%Y-%m-%d %H:%M", time.localtime(current_timestamp))
    print(f"Capture {i + 1}/{NUM_CAPTURES} saved ({readable})")

print(len(cover.retrieve()))