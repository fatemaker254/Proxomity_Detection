import cv2
import numpy as np
import torch
import time
import random


FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WARNING_DISTANCE = 250
DANGER_DISTANCE = 120

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_TRACKED = (0, 255, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX

VIRTUAL_TARGET_POS = (
    random.randint(70, FRAME_WIDTH - 70),
    random.randint(70, FRAME_HEIGHT - 70)
)
TARGET_RADIUS = 30

#Loading model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
model.conf = 0.45  # confidence threshold


def detect_hand_centroid(frame):
    results = model(frame[..., ::-1], size=416)  # BGR->RGB
    detections = results.xyxy[0]

    if len(detections) == 0:
        return None

    # Best detection = highest confidence
    best = detections[0]
    x1, y1, x2, y2, conf, cls = best.tolist()
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return (cx, cy), (int(x1), int(y1), int(x2), int(y2))


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        hand_data = detect_hand_centroid(frame)

        current_state = "SAFE"
        state_color = COLOR_GREEN
        distance = float("inf")

        # Draw proximity zones + target
        cv2.circle(frame, VIRTUAL_TARGET_POS, WARNING_DISTANCE, (0,150,150), 2)
        cv2.circle(frame, VIRTUAL_TARGET_POS, DANGER_DISTANCE, (0,0,150), 3)
        cv2.circle(frame, VIRTUAL_TARGET_POS, TARGET_RADIUS, COLOR_GREEN, -1)

        if hand_data:
            (cx, cy), (x1, y1, x2, y2) = hand_data

            # Draw box + centroid
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_TRACKED, 2)
            cv2.circle(frame, (cx,cy), 10, COLOR_TRACKED, -1)

            dx = cx - VIRTUAL_TARGET_POS[0]
            dy = cy - VIRTUAL_TARGET_POS[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= DANGER_DISTANCE:
                current_state = "DANGER"
                state_color = COLOR_RED
            elif distance <= WARNING_DISTANCE:
                current_state = "WARNING"
                state_color = COLOR_YELLOW

        # HUD
        cv2.putText(frame, f"State: {current_state}", (10,30), FONT, 0.8, state_color, 2)
        if distance != float("inf"):
            cv2.putText(frame, f"Dist: {distance:.1f}", (10,65), FONT, 0.7, state_color, 2)

        if current_state == "DANGER":
            cv2.putText(frame, "DANGER DANGER",
                        (FRAME_WIDTH//2 - 130, FRAME_HEIGHT//2),
                        FONT, 1.5, COLOR_RED, 4)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 120,30),
                    FONT, 0.7, COLOR_GREEN,2)

        cv2.imshow("Hand Proximity with YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
