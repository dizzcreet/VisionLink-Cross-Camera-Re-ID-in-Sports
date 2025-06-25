from ultralytics import YOLO

model = YOLO("models/best.pt")

def detect_players(frame):
    results = model(frame)[0]
    return results.boxes
