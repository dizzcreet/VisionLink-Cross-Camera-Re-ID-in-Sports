def extract_features(frame, boxes):
    features = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        features.append({"coords": center, "image": crop})
    return features
