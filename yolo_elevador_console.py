import time
import cv2
from ultralytics import YOLO

# -------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------

CAM_INDEX = 0          # índice de cámara (ajusta si hace falta)
CONF_THRES = 0.5       # umbral de confianza mínimo
MODEL_PATH = "yolov8n.pt"

# -------------------------------------------------------
# Lógica del elevador
# -------------------------------------------------------

def decidir_accion(num_personas: int) -> str:
    """
    Lógica:
      - 0 personas  -> ABRIR puerta y NO subir.
      - 1 a 3 pers. -> CERRAR puerta y SUBIR.
      - >3 personas -> NO cerrar, NO subir.
    """
    if num_personas == 0:
        return "abrir_no_subir"
    elif 1 <= num_personas <= 3:
        return "cerrar_subir"
    else:
        return "no_cerrar_no_subir"


# -------------------------------------------------------
# Procesado de un frame (sin imshow)
# -------------------------------------------------------

def procesar_frame(frame, model):
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

    accion = decidir_accion(num_personas)
    return num_personas, accion


# -------------------------------------------------------
# Main: detección continua sin ventanas
# -------------------------------------------------------

def main():
    print("===========================================")
    print("  SISTEMA ELEVADOR - YOLO (modo consola)")
    print("===========================================")
    print("Reglas:")
    print("  - 0 personas   -> ABRIR puerta, NO subir")
    print("  - 1 a 3 pers.  -> CERRAR puerta y SUBIR")
    print("  - >3 personas  -> NO cerrar, NO subir")
    print("-------------------------------------------")
    print("Controles:")
    print("  - CTRL + C para detener la ejecución")
    print("===========================================")

    print("[INFO] Cargando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Modelo cargado.")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con índice {CAM_INDEX}.")
        return

    ultima_accion = None
    ultimo_num = None
    t_ultimo_print = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer frame de la cámara.")
                break

            num_personas, accion = procesar_frame(frame, model)

            # Solo imprimir si cambió algo o cada cierto tiempo
            ahora = time.time()
            if (num_personas != ultimo_num) or (accion != ultima_accion) or (ahora - t_ultimo_print > 3.0):
                print("-------------------------------------------")
                print(f"Personas detectadas: {num_personas}")
                if accion == "abrir_no_subir":
                    print("ACCIÓN: ABRIR puerta, NO subir.")
                    # Aquí iría la señal al actuador para abrir puerta
                elif accion == "cerrar_subir":
                    print("ACCIÓN: CERRAR puerta y SUBIR elevador.")
                    # Aquí iría la señal al actuador para cerrar y subir
                else:
                    print("ACCIÓN: NO cerrar puerta, NO subir (sobreocupado).")
                    # Aquí iría la lógica de bloqueo del elevador

                ultima_accion = accion
                ultimo_num = num_personas
                t_ultimo_print = ahora

            # Pequeño sleep para no saturar CPU
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por usuario (CTRL+C). Saliendo...")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
