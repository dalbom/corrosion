import cv2

img = cv2.imread("imageData_S11_S21_Ph11.png")

cv2.imwrite("imageData_S11_S21_Ph11_remap.png", (img-127)*2)
