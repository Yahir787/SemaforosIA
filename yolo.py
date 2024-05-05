import cv2
import torch

def load_model():
    """ Cargar el modelo preentrenado de YOLOv5 """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detector(cap: object):
    model = load_model()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        preds = model(frame)
        print(preds)

    cap.release()

if __name__ == "__main__":
    cap = cv2.VideoCapture("data/video1.mp4")
    detector(cap)