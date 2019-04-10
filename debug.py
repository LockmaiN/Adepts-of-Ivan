import cv2
import face_detect_oop as fdo


test = fdo.FaceDetector('haarcascade_frontalface_default.xml', 0)

while True:


    cv2.imshow('Video', test.analyze())
    cv2.imshow('test', test.get_face(test.analyze()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()