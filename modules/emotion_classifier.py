import cv2


class EmotionClassifier:
    def __init__(self, model_file):
        self.face_emotion = cv2.face.FisherFaceRecognizer_create()
        self.face_emotion.read(model_file)
        self.emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

    def get_emotion(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (350, 350))
        emotion_prediction = self.face_emotion.predict(gray_image)
        return self.emotions[emotion_prediction[0]]
