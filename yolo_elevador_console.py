import cv2
import time
from ultralytics import YOLO

# -------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------

CAM_INDEX = 0
CONF_THRES = 0.5
DECISION_INTERVAL = 5.0   # segundos entre decisiones

model = YOLO("yolov8n.pt")

# -------------------------------------------------------
# Lógica del elevador
# -------------------------------------------------------

def decidir_accion(num_personas):
    if num_personas == 0:
        return "abrir_no_subir"
    elif 1 <= num_personas <= 3:
        return "cerrar_subir"
    else:
        return "no_cerrar_no_subir"

# -------------------------------------------------------
# Procesar frame
# -------------------------------------------------------

def procesar_frame(frame):
    results = model(frame, verbose=False)
    r = results[0]

    num_personas = 0

    boxes = r.boxes.xyxy
    clases = r.boxes.cls
    confs = r.boxes.conf

    if boxes is not None and len(boxes) > 0:
        for box, cls, conf in zip(boxes, clases, confs):
            cls_id = int(cls.item())
            conf_val = float(conf.item())

            if cls_id == 0 and conf_val >= CONF_THRES:
                num_personas += 1
                x1, y1, x2, y2 = box.int().tolist()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"person {conf_val:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 1)

    return num_personas

# -------------------------------------------------------
# Main loop: decisión automática cada 5s
# -------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {CAM_INDEX}")
        return

    print("YOLO elevador:")
    print(" - Decisiones automáticas cada 5 segundos")
    print(" - ESC para salir")

    cv2.namedWindow("Elevador YOLO")

    last_decision_time = time.time()
    last_action = "N/A"
    last_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame")
            break

        # Procesar detección
        num_personas = procesar_frame(frame)

        # Tomar decisión cada DECISION_INTERVAL
        now = time.time()
        if now - last_decision_time >= DECISION_INTERVAL:
            last_action = decidir_accion(num_personas)
            last_count = num_personas

            print("-------------------------------------")
            print(f"Personas detectadas: {num_personas}")
            print(f"Acción: {last_action}")

            last_decision_time = now

        # Mostrar info en la ventana
        cv2.putText(frame,
                    f"Personas (ahora): {num_personas}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2)

        cv2.putText(frame,
                    f"Ultima accion: {last_action}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

        cv2.putText(frame,
                    f"Siguiente decision en: {int(DECISION_INTERVAL - (now - last_decision_time))}s",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

        cv2.imshow("Elevador YOLO", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
