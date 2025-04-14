import cv2
from ultralytics import YOLO

model = YOLO("")

cap = cv2.VideoCapture(0) # 0/1/-1

while True:
        success, frame = cap.read()

        if not success:
            print("Камера не работает")
            break
        
        results = model(frame, conf=0.5)

        for result in results
            for box in result.boxes:
                x1,y1,x2,y2 = [int(i) for i in box.xyxy[0]]
                cv2.rectangle(frame, (x1,y1), (x2,y2),(0, 255, 0), 2)
                cv2.putText(frame, "", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("", frame)

        if cv2.waitKey(1) & 0xFF == ord('q')
            break
cap.release()
cv2.destroyAllWindows()