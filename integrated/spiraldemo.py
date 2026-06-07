import cover
import cv2
from slowlampthree import drawSpiral

currentData = cover.retrieve()
spiralImage = drawSpiral(currentData)
cv2.imwrite("spiral.png", spiralImage)