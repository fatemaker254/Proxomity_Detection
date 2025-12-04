## Hand Proximity Safety Detection System without MediaPipe, OpenPose, cloud AI APIs

Real-time Hand Tracking & Danger Detection using OpenCV + PyTorch(offline dependencies)

## Project Overview

This project demonstrates a real-time safety monitoring system using a webcam.  
A virtual target is displayed on the screen and the system tracks the user’s hand.  
When the hand approaches or touches the virtual area, a visual warning is activated:

| State | Condition | UI Feedback |
|-------|----------|-------------|
| SAFE | Hand far from target | Green status |
| WARNING | Hand approaching target | Yellow status |
| DANGER | Hand very close / touching | Red flashing alert + "DANGER DANGER" |

This model is capable of tracking both fingers or any part of the hand and checks at >20 FPS in real-time and uses Python + OpenCV + NumPy + PyTorch.
The Yolov5n small model is custom trained with just 511 images, making it small and effective.

## Technology Stack

| Component | Library / Method |
|----------|-----------------|
| Hand Detection | YOLOv5n (trained on custom dataset) |
| Fingertip Localization | Convex Hull + Topmost point |
| Proximity Logic | Euclidean distance from virtual target |
| Real-time Webcam Feed | OpenCV |
| Visualization | Live overlays + Danger Alerts |


## Instalation Procedure
1. Clone the Repo
2. python -m venv venv
3. venv\Scripts\activate     # Creating virtual environment
4. pip install -r requirements.txt
5. python main.py # to run the model and get the video capture mode onn 

Sample video: 
<video controls src="Sample_video.mp4" title="Title"></video>