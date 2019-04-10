import cv2
import sys
import numpy as np


class FaceDetector:
    def __init__(self, Cascade, Camera):
        self.faceCascade = cv2.CascadeClassifier(Cascade)
        self.video_capture = cv2.VideoCapture(Camera)

    def analyze(self):
        ret, frame = self.video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (176, 25, 143), 2)
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            return frame
        pass

    def get_face(self, frame):
        rows, cols, ch = frame.shape
        pts1 = np.float32([[self.x, self.y], [self.x + self.w, self.y], [self.x, self.y + self.h], [self.x + self.w, self.y + self.h]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        face = cv2.warpPerspective(frame, M, (300, 300))
        return face




