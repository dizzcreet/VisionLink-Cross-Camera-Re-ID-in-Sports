import cv2
from utils.detection import detect_players
from utils.feature_extraction import extract_features
from utils.matching import match_players
from utils.visualization import draw_boxes

video1 = cv2.VideoCapture("videos/broadcast.mp4")
video2 = cv2.VideoCapture("videos/tacticam.mp4")

frame_width = int(video1.get(3))
frame_height = int(video1.get(4))

# Optional: Save output video
save_output = False
if save_output:
    out1_writer = cv2.VideoWriter('output_broadcast.mp4',
                                  cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                  (frame_width, frame_height))
    out2_writer = cv2.VideoWriter('output_tacticam.mp4',
                                  cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                  (frame_width, frame_height))

frame_count = 0
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 or not ret2:
        break

    # --- DETECTION ---
    boxes1 = detect_players(frame1)
    boxes2 = detect_players(frame2)

    # --- FEATURE EXTRACTION ---
    features1 = extract_features(frame1, boxes1)
    features2 = extract_features(frame2, boxes2)

    # --- MATCHING ---
    matches = match_players(features1, features2)

    # --- ID ASSIGNMENT (basic one-to-one id) ---
    id_map1 = {i: i for i, _ in matches}
    id_map2 = {j: i for i, j in matches}

    # --- VISUALIZATION ---
    vis_frame1 = draw_boxes(frame1, boxes1, id_map1)
    vis_frame2 = draw_boxes(frame2, boxes2, id_map2)

    # --- DISPLAY ---
    cv2.imshow("Broadcast View", vis_frame1)
    cv2.imshow("Tacticam View", vis_frame2)

    if save_output:
        out1_writer.write(vis_frame1)
        out2_writer.write(vis_frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# --- CLEAN UP ---
video1.release()
video2.release()
if save_output:
    out1_writer.release()
    out2_writer.release()

cv2.destroyAllWindows()
