import cv2
import os
import uuid

DATA_DIR = 'dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

HAND_PATH = os.path.join(DATA_DIR, 'hand')
NOHAND_PATH = os.path.join(DATA_DIR, 'not_hand')
os.makedirs(HAND_PATH, exist_ok=True)
os.makedirs(NOHAND_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count_hand = count_nohand = 0

print("Press 'h' to save hand image, 'n' to save not-hand image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.putText(frame, "Press H=hand | N=not-hand | Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Data Capture', frame)
    
    key = cv2.waitKey(1)

    if key == ord('h'):
        img_name = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(HAND_PATH, img_name), frame)
        count_hand += 1
        print("Hand image saved:", count_hand)

    elif key == ord('n'):
        img_name = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(NOHAND_PATH, img_name), frame)
        count_nohand += 1
        print("Not-hand image saved:", count_nohand)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
