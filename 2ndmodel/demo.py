import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hand_cnn_model import SmallHandCNN
import time
import random

# -------------------------
# 1. Load model (CPU only)
# -------------------------
device = torch.device("cpu")
model = SmallHandCNN().to(device)
model.load_state_dict(torch.load("hand_cnn.pth", map_location=device))
model.eval()

# -------------------------
# 2. Helper functions
# -------------------------

def preprocess_crop_for_cnn(crop_bgr):
    """Convert BGR crop to normalized tensor (1,3,64,64)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (64, 64))
    tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)

def is_hand(crop_bgr, threshold=0.7):
    """Use CNN to decide if crop is hand."""
    with torch.no_grad():
        x = preprocess_crop_for_cnn(crop_bgr)
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return prob > threshold, prob

def find_hand_candidate(frame):
    """
    Use skin-color segmentation + contour to propose a hand region.
    Returns (x, y, w, h, cx, cy) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (tune if needed)
    lower = np.array([0, 40, 40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 1500:  # ignore tiny blobs
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w // 2
    cy = y + h // 2
    return x, y, w, h, cx, cy

def pick_random_point(w, h, margin=60):
    """Random virtual point inside frame with margin."""
    return random.randint(margin, w - margin), random.randint(margin, h - margin)

def compute_state(distance, safe_thresh, danger_thresh):
    if distance <= danger_thresh:
        return "DANGER"
    elif distance <= safe_thresh:
        return "WARNING"
    else:
        return "SAFE"

# -------------------------
# 3. Main real-time loop
# -------------------------

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam")
        return

    h, w, _ = frame.shape

    # Initial virtual object (point)
    px, py = pick_random_point(w, h)

    # Distance thresholds as functions of image diagonal
    diag = np.sqrt(w**2 + h**2)
    SAFE_THRESH = 0.40 * diag      # SAFE when > this
    DANGER_THRESH = 0.15 * diag    # DANGER when <= this

    prev_time = time.time()
    danger_flash = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        display = frame.copy()

        state = "NO HAND"
        hand_prob = 0.0
        point_color = (0, 255, 0)   # default green

        hand_info = find_hand_candidate(frame)
        if hand_info is not None:
            x, y, wb, hb, cx, cy = hand_info

            crop = frame[y:y+hb, x:x+wb]
            is_hand_flag, hand_prob = is_hand(crop)

            if is_hand_flag:
                # Draw bounding box and centroid
                cv2.rectangle(display, (x, y), (x + wb, y + hb), (255, 0, 0), 2)
                cv2.circle(display, (cx, cy), 6, (255, 0, 0), -1)

                # Distance from hand to virtual point
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                state = compute_state(dist, SAFE_THRESH, DANGER_THRESH)

                # Color of virtual object based on state
                if state == "SAFE":
                    point_color = (0, 255, 0)       # green
                elif state == "WARNING":
                    point_color = (0, 255, 255)     # yellow
                elif state == "DANGER":
                    point_color = (0, 0, 255)       # red

                # Draw connecting line
                cv2.line(display, (cx, cy), (px, py), point_color, 2)

                # If we hit DANGER, respawn the virtual object elsewhere
                if state == "DANGER":
                    px, py = pick_random_point(w, h)
            else:
                state = "NOT HAND (CNN)"

        # Draw virtual object (always visible)
        cv2.circle(display, (px, py), 10, point_color, -1)

        # FPS
        end = time.time()
        fps = 1.0 / (end - start + 1e-6)

        # State text
        cv2.putText(display, f"State: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, point_color, 2)

        cv2.putText(display, f"Hand prob: {hand_prob:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(display, f"FPS: {fps:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # DANGER DANGER overlay (flashing)
        if state == "DANGER":
            danger_flash = not danger_flash
            if danger_flash:
                cv2.putText(display, "DANGER DANGER", (int(w * 0.20), int(h * 0.55)),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)

        cv2.imshow("Hand-Object Interaction (SAFE / WARNING / DANGER)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
