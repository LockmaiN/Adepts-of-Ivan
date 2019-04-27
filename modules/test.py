import cv2
from pathlib import Path
from configparser import ConfigParser

from modules.face_detection import FaceDetector
from modules.emotion_classifier import EmotionClassifier

config_path = str(Path(__file__).parents[1] / 'config.conf')
config = ConfigParser()
config.read(config_path)

font = cv2.FONT_HERSHEY_SIMPLEX

face_detector = FaceDetector(str(Path(__file__).parents[1] / config.get('models', 'face_detection')))
emotion_classifier = EmotionClassifier(str(Path(__file__).parents[1] / config.get('models', 'emotion_classifier')))

vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()

    faces = face_detector.get_face(frame)
    if len(faces) == 0:
        cv2.imshow('result', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion = emotion_classifier.get_emotion(frame[y: (y + h), x: (x + h), :])
        cv2.putText(frame, emotion, (x, y), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('result', frame)
        if cv2.waitKey(1) == ord('q'):
            break
vc.release()
cv2.destroyAllWindows()
