import cv2
import numpy as np
import torch

# Cargar el modelo preentrenado de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Coordenadas aproximadas del polígono basadas en la imagen proporcionada
# Reemplazar con las coordenadas reales obtenidas de tu video
##borde inferior izquierdo, borde superior izquierdo, borde inferior derecho, borde superior derecho
detection_area = [(880, 200), (615, 750), (1370, 750), (1080, 200)]

def point_inside_polygon(x, y, poly):
    """ Verifica si un punto está dentro de un polígono. """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y) * (p2x-p1x) / (p2y-p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def draw_poly(frame, poly):
    """ Dibuja un polígono en el frame """
    pts = np.array(poly, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

def detector():
    cap = cv2.VideoCapture("data/video1.mp4")  # Cambiar al video de entrada
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dibuja el área de detección
        draw_poly(frame, detection_area)

        # Realiza detección usando YOLOv5
        results = model(frame)
        detections = results.xyxy[0].numpy()

        # Filtra detecciones por clase de vehículo y confianza
        vehicles = [det for det in detections if det[5] in [2, 3, 5, 7] and det[4] > 0.4]

        # Contar vehículos dentro de la región de detección
        vehicle_count = 0
        for *xyxy, conf, cls in vehicles:
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            if point_inside_polygon(x_center, y_center, detection_area):
                vehicle_count += 1
                # Dibuja los bounding boxes de los vehículos detectados dentro de la región
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        # Si hay más de 4 vehículos, agrega tiempo al semáforo
        if vehicle_count > 4:
            print("Más de 4 vehículos detectados. Agregando tiempo...")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector()
