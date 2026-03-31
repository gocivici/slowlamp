import cover
import cv2
from spiral import drawSpiral

currentData = cover.retrieve()
spiralImage = drawSpiral(currentData)
cv2.imwrite("spiral.png", spiralImage)