import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

# -------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------

CAM_INDEX = 0
model = YOLO("yolov8n.pt")
CONF_THRES = 0.5

# -------------------------------------------------------
# CONFIGURACIÓN SERVOS SG90
# -------------------------------------------------------

SERVO_PUERTA_PIN   = 17  # GPIO17 pin 11
SERVO_ELEVADOR_PIN = 27  # GPIO27 pin 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PUERTA_PIN, GPIO.OUT)
GPIO.setup(SERVO_ELEVADOR_PIN, GPIO.OUT)

# PWM a 50 Hz
servo_puerta_pwm   = GPIO.PWM(SERVO_PUERTA_PIN, 50)
servo_elevador_pwm = GPIO.PWM(SERVO_ELEVADOR_PIN, 50)

servo_puerta_pwm.start(0)
servo_elevador_pwm.start(0)

# Ángulos configurables
ANGULO_ABRIR_PUERTA   = 90
ANGULO_CERRAR_PUERTA  = 0

ANGULO_NO_SUBIR       = 0
ANGULO_SUBIR          = 90

def mover_servo(pwm, angle):
    duty = 2 + (angle / 18.0)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.35)
    pwm.ChangeDutyCycle(0)

def ejecutar_accion_servos(accion):
    """
    - abrir_no_subir      -> puerta abierta, elevador detenido
    - cerrar_subir        -> puerta cerrada, elevador sube
    - no_cerrar_no_subir  -> puerta NO se mueve, elevador detenido
    """

    if accion == "abrir_no_subir":
        mover_servo(servo_puerta_pwm, ANGULO_ABRIR_PUERTA)
        mover_servo(servo_elevador_pwm, ANGULO_NO_SUBIR)

    elif accion == "cerrar_subir":
        mover_servo(servo_puerta_pwm, ANGULO_CERRAR_PUERTA)
        mover_servo(servo_elevador_pwm, ANGULO_SUBIR)

    elif accion == "no_cerrar_no_subir":
        mover_servo(servo_elevador_pwm, ANGULO_NO_SUBIR)
        # puerta queda donde esté


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
        color_accion = (255, 255, 0)

    elif accion == "cerrar_subir":
        texto_accion = "ACCION: CERRAR puerta y SUBIR"
        color_accion = (0, 255, 0)

    else:
        texto_accion = "ACCION: NO cerrar puerta, NO subir"
        color_accion = (0, 0, 255)

    cv2.putText(frame, texto_personas, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, texto_accion, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_accion, 2)

    return frame, num_personas, accion


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        GPIO.cleanup()
        return

    cv2.namedWindow("Elevador YOLO")
    modo_resultado = False
    frame_resultado = None
    frame = None

    try:
        while True:
            if not modo_resultado:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, "EN VIVO - ESPACIO: evaluar, ESC: salir",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                cv2.imshow("Elevador YOLO", frame)

            else:
                frame_vis = frame_resultado.copy()
                cv2.putText(frame_vis, "RESULTADO - ENTER/ESPACIO: volver, ESC: salir",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                cv2.imshow("Elevador YOLO", frame_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Saliendo...")
                break

            if not modo_resultado and key == 32:
                if frame is not None:
                    frame_capt = frame.copy()
                    frame_resultado, num_personas, accion = procesar_frame(frame_capt)

                    print(f"Captura -> personas: {num_personas}, accion: {accion}")

                    ejecutar_accion_servos(accion)
                    modo_resultado = True

            elif modo_resultado and (key == 13 or key == 32):
                modo_resultado = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_puerta_pwm.stop()
        servo_elevador_pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()

