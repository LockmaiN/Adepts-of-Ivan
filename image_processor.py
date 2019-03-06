import cv2


def img_process(image, EdgesEnabledTrue):
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (256, 256))
    if EdgesEnabledTrue == 1:
        img = cv2.Canny(img, 100, 200)
    return img
