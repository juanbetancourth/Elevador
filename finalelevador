import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

# -------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------

# Índice de la cámara:
#  - 0 suele ser la cámara interna del portátil
#  - 1, 2, ... suelen ser cámaras USB
CAM_INDEX = 0  # cámbialo a 1 para probar tu cámara USB

# Modelo YOLO preentrenado
model = YOLO("yolov8n.pt")

# Umbral de confianza mínimo para contar una detección como persona
CONF_THRES = 0.5

# -------------------------------------------------------
# CONFIGURACIÓN SERVO PUERTA (SG90 en GPIO17)
# -------------------------------------------------------
SERVO_PUERTA_PIN = 17  # GPIO17, pin físico 11

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PUERTA_PIN, GPIO.OUT)

# Frecuencia típica para servos: 50 Hz
servo_puerta_pwm = GPIO.PWM(SERVO_PUERTA_PIN, 50)
servo_puerta_pwm.start(0)

# Ángulos de referencia (ajusta si tu mecánica es al revés)
ANGULO_ABRIR_PUERTA = 90   # puerta abierta
ANGULO_CERRAR_PUERTA = 0   # puerta cerrada

def mover_servo_puerta(angle):
    """Mueve el servo de la puerta al ángulo indicado."""
    duty = 2 + (angle / 18.0)
    servo_puerta_pwm.ChangeDutyCycle(duty)
    time.sleep(0.35)
    servo_puerta_pwm.ChangeDutyCycle(0)

def ejecutar_accion_servos(accion):
    """
    Ejecuta el movimiento de la puerta según la acción del elevador.
    """
    if accion == "abrir_no_subir":
        # Abrir puerta
        mover_servo_puerta(ANGULO_ABRIR_PUERTA)

    elif accion == "cerrar_subir":
        # Cerrar puerta
        mover_servo_puerta(ANGULO_CERRAR_PUERTA)

    # Para "no_cerrar_no_subir" no movemos el servo, se mantiene en su posición
    # elif accion == "no_cerrar_no_subir":
    #     pass
    # Si luego agregas más acciones, las manejas aquí.


# -------------------------------------------------------
# Lógica del elevador
# -------------------------------------------------------
def decidir_accion(num_personas):
    """
    Lógica:
      - 0 personas -> ABRIR puerta y NO subir.
      - 1 a 3 personas -> CERRAR puerta y SUBIR.
      - más de 3 personas -> NO cerrar, NO subir.
    """
    if num_personas == 0:
        accion = "abrir_no_subir"
    elif 1 <= num_personas <= 3:
        accion = "cerrar_subir"
    else:
        accion = "no_cerrar_no_subir"
    return accion

# -------------------------------------------------------
# Ejecutar detección y decisión sobre un frame concreto
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

            # COCO: clase 0 = persona
            if cls_id == 0 and conf_val >= CONF_THRES:
                num_personas += 1
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf_val:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

    accion = decidir_accion(num_personas)

    texto_personas = f"Personas: {num_personas}"

    if accion == "abrir_no_subir":
        texto_accion = "ACCION: ABRIR puerta, NO subir"
        color_accion = (255, 255, 0)  # amarillo
    elif accion == "cerrar_subir":
        texto_accion = "ACCION: CERRAR puerta y SUBIR"
        color_accion = (0, 255, 0)    # verde
    else:
        texto_accion = "ACCION: NO cerrar puerta, NO subir"
        color_accion = (0, 0, 255)    # rojo

    cv2.putText(
        frame,
        texto_personas,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        texto_accion,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_accion,
        2,
    )

    return frame, num_personas, accion

# -------------------------------------------------------
# Main: UNA sola ventana, con modos EN VIVO / RESULTADO
# -------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {CAM_INDEX}.")
        print("Prueba con otros índices (0, 1, 2...).")
        GPIO.cleanup()
        return

    print("YOLO elevador:")
    print("  - Vista en vivo")
    print("  - Presiona ESPACIO para capturar y decidir")
    print("  - Presiona ENTER o ESPACIO para volver al modo en vivo después de una captura")
    print("  - Presiona ESC para salir")

    cv2.namedWindow("Elevador YOLO")

    # Modo:
    #   False -> EN VIVO
    #   True  -> mostrándose el último resultado procesado
    modo_resultado = False
    frame_resultado = None
    frame = None  # último frame leído en vivo

    try:
        while True:
            if not modo_resultado:
                # Modo en vivo: leer la cámara
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer frame de la cámara.")
                    break

                cv2.putText(
                    frame,
                    "EN VIVO - ESPACIO: evaluar, ESC: salir",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Elevador YOLO", frame)

            else:
                # Modo resultado: mostrar el último frame procesado
                frame_vis = frame_resultado.copy()
                cv2.putText(
                    frame_vis,
                    "RESULTADO - ENTER/ESPACIO: volver, ESC: salir",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Elevador YOLO", frame_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Saliendo...")
                break

            # ESPACIO en modo en vivo -> capturar y procesar
            if not modo_resultado and key == 32:
                if frame is not None:
                    frame_capturado = frame.copy()
                    frame_resultado, num_personas, accion = procesar_frame(frame_capturado)
                    print(f"Captura -> personas: {num_personas}, accion: {accion}")

                    # Aquí se mueve el servo según la acción
                    ejecutar_accion_servos(accion)

                    modo_resultado = True

            # ENTER (13) o ESPACIO (32) en modo resultado -> volver a EN VIVO
            elif modo_resultado and (key == 13 or key == 32):
                modo_resultado = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_puerta_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
