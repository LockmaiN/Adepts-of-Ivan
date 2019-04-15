import cv2


class FaceDetector:
    def __init__(self, model_file):
        self.faceCascade = cv2.CascadeClassifier(model_file)

    def get_face(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return faces
