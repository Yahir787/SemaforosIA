import cv2
import numpy as np
import torch
import tkinter as tk
from threading import Thread, Lock, Event
import time

# Carga el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

detection_area = [(880, 200), (615, 750), (1370, 750), (1080, 200)]

def point_inside_polygon(x, y, poly):
    #Verifica si un punto está dentro de un polígono. 
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
    # Dibuja un polígono en el frame 
    pts = np.array(poly, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

# Función para procesar el video y detectar vehículos
def process_video(semaphore_state, vehicle_count_lock, stop_event, model):
    cap = cv2.VideoCapture("data/video1.mp4")  # Cambiar al video de entrada
    
    while cap.isOpened() and not stop_event.is_set():
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

        with vehicle_count_lock:
            semaphore_state["vehicle_count"] = vehicle_count

        print(vehicle_count)
        
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Función para manejar la lógica de semáforos
def semaphore_logic(semaphore_labels, semaphore_state, vehicle_count_lock, stop_event, canvas1, canvas2, canvas3, canvas4, canvas5, canvas6):
    semaphore1_time = 20  # Semáforo 1 inicia en verde por 20 segundos
    semaphore1_state = "green"  # Estado actual del semáforo 1
    semaphore1_extra_count = 0  # Contador de aumentos extras para el semáforo 1
    semaphore2_time = 23  # Semáforo 2 inicia en rojo por 20 segundos
    semaphore2_state = "red"  # Estado actual del semáforo 2
    semaphore3_time = 20
    semaphore3_state = "green"
    semaphore4_time = 23  # Semáforo 4 inicia en rojo por 20 segundos
    semaphore4_state = "red"  # Estado actual del semáforo 4
    semaphore5_time = 23  # Semáforo 5 inicia en rojo por 20 segundos
    semaphore5_state = "red"  # Estado actual del semáforo 5
    semaphore6_time = 20
    semaphore6_state = "green"
    yellow_duration = 3  # Duración del amarillo

    while not stop_event.is_set():
        with vehicle_count_lock:
            vehicle_count = semaphore_state["vehicle_count"]

        if vehicle_count > 1 and semaphore1_time <= 5 and semaphore1_state == "green" and semaphore1_extra_count < 3:
            extra_time = 5
            semaphore1_extra_count += 1
        else:
            extra_time = 0

        semaphore1_time -= 1
        semaphore2_time -= 1
        semaphore3_time -= 1
        semaphore4_time -= 1
        semaphore5_time -= 1
        semaphore6_time -= 1

        semaphore1_time += extra_time
        semaphore2_time += extra_time
        semaphore3_time += extra_time
        semaphore4_time += extra_time
        semaphore5_time += extra_time
        semaphore6_time += extra_time

        if semaphore1_time <= 0:
            if semaphore1_state == "green":
                semaphore1_state = "yellow"
                semaphore3_state = "yellow"
                semaphore6_state = "yellow"
                semaphore1_time = yellow_duration
                semaphore3_time = yellow_duration
                semaphore6_time = yellow_duration
            elif semaphore1_state == "yellow":
                semaphore1_state = "red"
                semaphore1_time = 23
                semaphore3_state = "red"
                semaphore3_time = 23
                semaphore6_state = "red"
                semaphore6_time = 23
                semaphore2_state = "green"
                semaphore4_state = "green"
                semaphore5_state = "green"
                semaphore2_time = 20
                semaphore4_time = 20
                semaphore5_time = 20
            else:
                semaphore1_state = "green"
                semaphore1_time = 20
                semaphore3_state = "green"
                semaphore3_time = 20
                semaphore6_state = "green"
                semaphore6_time = 20
                semaphore1_extra_count = 0

        if semaphore2_time <= 0:
            if semaphore2_state == "green":
                semaphore2_state = "yellow"
                semaphore2_time = yellow_duration
            elif semaphore2_state == "yellow":
                semaphore2_state = "red"
                semaphore2_time = 23

        if semaphore4_time <= 0:
            if semaphore4_state == "green":
                semaphore4_state = "yellow"
                semaphore4_time = yellow_duration
            elif semaphore4_state == "yellow":
                semaphore4_state = "red"
                semaphore4_time = 23

        if semaphore5_time <= 0:
            if semaphore5_state == "green":
                semaphore5_state = "yellow"
                semaphore5_time = yellow_duration
            elif semaphore5_state == "yellow":
                semaphore5_state = "red"
                semaphore5_time = 23

        # Actualiza el color de los círculos según el estado actual de cada semáforo
        for i, color in enumerate(["red", "yellow", "green"]):
            canvas1.itemconfig(semaphore_labels["semaphore1"]["visuals"][i], fill=color if color == semaphore1_state else "gray")
            canvas2.itemconfig(semaphore_labels["semaphore2"]["visuals"][i], fill=color if color == semaphore2_state else "gray")
            canvas3.itemconfig(semaphore_labels["semaphore3"]["visuals"][i], fill=color if color == semaphore3_state else "gray")
            canvas4.itemconfig(semaphore_labels["semaphore4"]["visuals"][i], fill=color if color == semaphore4_state else "gray")
            canvas5.itemconfig(semaphore_labels["semaphore5"]["visuals"][i], fill=color if color == semaphore5_state else "gray")
            canvas6.itemconfig(semaphore_labels["semaphore6"]["visuals"][i], fill=color if color == semaphore6_state else "gray")

        semaphore_labels["semaphore1"]["timer"].config(text=f"{semaphore1_time}s ")
        semaphore_labels["semaphore2"]["timer"].config(text=f"{semaphore2_time}s ")
        semaphore_labels["semaphore3"]["timer"].config(text=f"{semaphore3_time}s ")
        semaphore_labels["semaphore4"]["timer"].config(text=f"{semaphore4_time}s ")
        semaphore_labels["semaphore5"]["timer"].config(text=f"{semaphore5_time}s ")
        semaphore_labels["semaphore6"]["timer"].config(text=f"{semaphore6_time}s ")
        semaphore_labels["vehicle_count"].config(text=f"Vehículos detectados: {vehicle_count}")

        time.sleep(1)



# Función para configurar la interfaz gráfica de usuario (GUI)
def setup_gui(semaphore_state):
    root = tk.Tk()
    root.title("Simulacion de Semaforos")

    # Marco y etiqueta para cada semáforo
    frame1 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label1 = tk.Label(frame1, text="Semáforo 1.1")
    label1.pack()
    frame1.pack(side=tk.LEFT, padx=10, pady=10)

    frame2 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label2 = tk.Label(frame2, text="Semáforo 1.3")
    label2.pack()
    frame2.pack(side=tk.LEFT, padx=10, pady=10)

    frame3 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label3 = tk.Label(frame3, text="Semáforo 1.2")
    label3.pack()
    frame3.pack(side=tk.LEFT, padx=10, pady=10)

    frame4 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label4 = tk.Label(frame4, text="Semáforo 2.2")
    label4.pack()
    frame4.pack(side=tk.LEFT, padx=10, pady=10)

    frame5 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label5 = tk.Label(frame5, text="Semáforo 3.1")
    label5.pack()
    frame5.pack(side=tk.LEFT, padx=10, pady=10)

    frame6 = tk.Frame(root, width=70, height=150, bg='#0a0a0a')
    label6 = tk.Label(frame6, text="Semáforo 2.1")
    label6.pack()
    frame6.pack(side=tk.LEFT, padx=10, pady=10)

   
    # Canvas para cada semáforo
    canvas1 = tk.Canvas(frame1, width=70, height=150, bg='#191919')
    canvas1.pack()
    canvas2 = tk.Canvas(frame2, width=70, height=150, bg='#191919')
    canvas2.pack()
    canvas3 = tk.Canvas(frame3, width=70, height=150, bg='#191919')
    canvas3.pack()
    canvas4 = tk.Canvas(frame4, width=70, height=150, bg='#191919')
    canvas4.pack()
    canvas5 = tk.Canvas(frame5, width=70, height=150, bg='#191919')
    canvas5.pack()
    canvas6 = tk.Canvas(frame6, width=70, height=150, bg='#191919')
    canvas6.pack()

    # Dibuja los círculos para cada semáforo
    semaphore1_visuals = [canvas1.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]
    semaphore2_visuals = [canvas2.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]
    semaphore3_visuals = [canvas3.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]
    semaphore4_visuals = [canvas4.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]
    semaphore5_visuals = [canvas5.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]
    semaphore6_visuals = [canvas6.create_oval(20, 10 + i * 40, 50, 40 + i * 40, fill="gray") for i in range(3)]

    semaphore_labels = {
        "semaphore1": {"timer": tk.Label(frame1), "visuals": semaphore1_visuals},
        "semaphore2": {"timer": tk.Label(frame2), "visuals": semaphore2_visuals},
        "semaphore3": {"timer": tk.Label(frame3), "visuals": semaphore3_visuals},
        "semaphore4": {"timer": tk.Label(frame4), "visuals": semaphore4_visuals},
        "semaphore5": {"timer": tk.Label(frame5), "visuals": semaphore5_visuals},
        "semaphore6": {"timer": tk.Label(frame6), "visuals": semaphore6_visuals},
        "vehicle_count": tk.Label(root, text="Vehículos detectados: 0")
    }

    semaphore_labels["semaphore1"]["timer"].pack()
    semaphore_labels["semaphore2"]["timer"].pack()
    semaphore_labels["semaphore3"]["timer"].pack()
    semaphore_labels["semaphore4"]["timer"].pack()
    semaphore_labels["semaphore5"]["timer"].pack()
    semaphore_labels["semaphore6"]["timer"].pack()
    semaphore_labels["vehicle_count"].pack(pady=10)

    vehicle_count_lock = Lock()
    stop_event = Event()

    Thread(target=process_video, args=(semaphore_state, vehicle_count_lock, stop_event, model)).start()
    Thread(target=semaphore_logic, args=(semaphore_labels, semaphore_state, vehicle_count_lock, stop_event, canvas1, canvas2, canvas3, canvas4, canvas5, canvas6)).start()

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    semaphore_state = {"time_left": 20, "vehicle_count": 0} 
    setup_gui(semaphore_state)
