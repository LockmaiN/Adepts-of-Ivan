import cv2
from pathlib import Path
from configparser import ConfigParser

from modules.face_detection import FaceDetector
from modules.emotion_classifier import EmotionClassifier
from modules.person_detector import PersonDetector

config_path = str(Path(__file__).parents[1] / 'config.conf')
config = ConfigParser()
config.read(config_path)

font = cv2.FONT_HERSHEY_SIMPLEX

face_detector = FaceDetector(str(Path(__file__).parents[1] / config.get('models', 'face_detection')))
emotion_classifier = EmotionClassifier(str(Path(__file__).parents[1] / config.get('models', 'emotion_classifier')))
object_detection = PersonDetector(str(Path(__file__).parents[1] / config.get('models', 'person_detection')))
vc = cv2.VideoCapture('/home/dsabodashko/Downloads/kitchen_single_person_cam_cut_2.mp4')

while True:
    is_ok, frame = vc.read()
    if not is_ok:
        break

    persons = object_detection.run_inference(frame)
    for (xp, yp, wp, hp) in persons:
        faces = face_detector.get_face(frame[yp: (yp + hp), xp: (xp + hp), :])
        cv2.rectangle(frame, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
        if len(faces) == 0:
            continue
        x, y, w, h = min(faces, key=lambda a: a[1])
        if y > (yp + 0.2 * hp):
            continue
        cv2.rectangle(frame, (x + xp, y + yp), (x + xp + w, y + yp + h), (0, 255, 0), 2)
        emotion = emotion_classifier.get_emotion(frame[(y + yp): (y + yp + h), (x + xp): (x + xp + h), :])
        cv2.putText(frame, emotion, (x + xp, y + yp), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('result', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()
