from feat.detector import Detector
from write_custom_tags import write_custom_tags


def emotion_detector(path):

    detector = Detector(
        face_model="img2pose",
        landmark_model="mobilefacenet",
        au_model='svm',
        emotion_model="fer",
        facepose_model="img2pose",
    )

    face_predictions = detector.detect_image(path)
    # Happiness detection
    if face_predictions['happiness'].max() > 0.5:
        # WRITE EXIF TAG AS HAPPY PERSON DETECTION
        write_custom_tags('Emotion', path, 'happiness')
