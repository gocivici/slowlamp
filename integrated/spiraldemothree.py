import cover
import cv2
from slowlampthree import generate_slowlamp_views


currentData = cover.retrieve()

img_24h, img_30d, img_all_time = generate_slowlamp_views(currentData)

# 3. Save them to the Pi's RAM disk for speed and to save SD card life
cv2.imwrite("view_24h.png", img_24h)
cv2.imwrite("view_30d.png", img_30d)
cv2.imwrite("view_all_time.png", img_all_time)