from ultralytics import YOLO
import cv2
import time
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame)
    end = time.time()
    fps = 1/(end-start)
    annotated = results[0].plot()
    cv2.putText(annotated, f"FPS: {fps:.2f}", (20,40),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("YOLO Real-Time", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
