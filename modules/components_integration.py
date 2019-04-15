import cv2
import numpy as np
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

vc = cv2.VideoCapture('/home/dsabodashko/Downloads/kitchen_single_person_cam_cut_2.mp4')
vc_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
vc_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
vc_fps = vc.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('/home/dsabodashko/Downloads/kitchen_single_person_cam_cut_2_res.mp4',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      vc_fps,
                      (vc_width, vc_height))

frames_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
for _ in range(int(frames_num)):
    is_ok, frame = vc.read()
    if not is_ok:
        break
    faces = face_detector.get_face(frame)
    if len(faces) == 0:
        # cv2.imshow('result', frame)
        out.write(frame.astype(np.uint8))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion = emotion_classifier.get_emotion(frame[y: (y + h), x: (x + h), :])
        cv2.putText(frame, emotion, (x, y), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('result', frame)
        out.write(frame.astype(np.uint8))
vc.release()
out.release()
cv2.destroyAllWindows()
