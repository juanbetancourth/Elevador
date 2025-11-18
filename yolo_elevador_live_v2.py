import cv2
from ultralytics import YOLO

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
                modo_resultado = True

        # ENTER (13) o ESPACIO (32) en modo resultado -> volver a EN VIVO
        elif modo_resultado and (key == 13 or key == 32):
            modo_resultado = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
