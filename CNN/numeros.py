import cv2, numpy as np
from tensorflow import keras  


modelo = keras.models.load_model("modelo_treinado.h5")

cap = cv2.VideoCapture(0)  #mude para 1/2 se tiver outra câmera
if not cap.isOpened():
    raise RuntimeError("Não consegui abrir a webcam.")

ROI = 220  #tamanho do quadrado central
ultima_pred, ultima_conf = "-", 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = ROI // 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

    #desenha quadrado (ROI)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
    cv2.putText(frame, "Coloque o digito aqui", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)

   
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)            # inverte preto/branco
    gray = cv2.resize(gray, (28, 28))       # 28x28
    arr = gray.astype("float32") / 255.0    # normaliza [0,1]
    arr = arr.reshape(1, 28, 28, 1)         # (1,28,28,1)

    #PREDIÇÃO
    pred = modelo.predict(arr, verbose=0)[0]
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))
    ultima_pred, ultima_conf = str(cls), conf

    #resultado
    texto = f"Pred: {ultima_pred} | conf: {ultima_conf:.2f} | 'q' para sair"
    cv2.putText(frame, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)

    cv2.imshow("MNIST Webcam (ROI)", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
