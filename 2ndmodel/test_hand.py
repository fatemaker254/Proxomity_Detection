import cv2
import torch
import torch.nn.functional as F
import numpy as np
from hand_cnn_model import SmallHandCNN

device = torch.device("cpu")

# Load trained model
model = SmallHandCNN().to(device)
model.load_state_dict(torch.load("hand_cnn.pth", map_location=device))
model.eval()

def preprocess(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (64, 64))
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5  # normalize [-1,1]
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    if prob > 0.5:
        label = "HAND"
        color = (0, 255, 0)
    else:
        label = "NOT HAND"
        color = (0, 0, 255)

    cv2.putText(frame, f"{label} ({prob:.2f})", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Hand Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
