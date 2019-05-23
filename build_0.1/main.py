import cv2
import service


person_detector = service.Detector('./../models/frozen_inference_graph.pb')
face_detector = service.FaceDetector('./../models/haarcascade_frontalface_default.xml')
emotion_classifier = service.EmotionClassifier('./../models/emotion_classifier_model.xml')

data = []
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.65
cap = cv2.VideoCapture(0)
run = True
print('VIDEO STREAM STARTED')

while run:
    _, frame = cap.read()
    boxes, scores, classes = person_detector.process_frame(frame)
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] >= threshold:
            box = boxes[i]
            if box == 0:
                cv2.imshow('EmphatoAI', frame)
                cv2.waitKey(1)
            else:
                # for (x, y, w, h) in box:
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                faces = face_detector.get_face(frame[box[0]: (box[0] + box[2]), box[1]: (box[1] + box[3]), :])
                for (a, b, c, d) in faces:
                    cv2.rectangle(frame[box[0]: (box[0] + box[2]), box[1]: (box[1] + box[3]), :], (a, b),
                                  (a + c, b + d), (255, 0, 0), 3)
                    emotion = emotion_classifier.get_emotion(frame[b: (b + d), a: (a + c), :])
                    data.append(emotion)
                    cv2.putText(frame, emotion, (a + box[0], b + box[1]), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    text = 'Found ' + str(i + 1) + ' persons with ' + str(len(faces)) + ' faces.'
                    cv2.putText(frame, text, (10, 30), font, 1, (0, 0, 255), 1)
                    cv2.imshow('EmphatoAI', frame)
                    if cv2.waitKey(1) == ord('q'):
                        run = False

print(data)
cap.release()
cv2.destroyAllWindows()

