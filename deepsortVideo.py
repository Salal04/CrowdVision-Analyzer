import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  # contains create_box_encoder


# ----------------- User params -----------------
VIDEO_PATH = "vid3.mp4"  # apna video path yahan
OUTPUT_PATH = "vid3(output).mp4"     # optional, processed video save karna ho
YOLO_MODEL = "yolov8n.pt"
REID_WEIGHTS = "deep_sort/networks/mars-small128.pb"
MASK_GLASSES_DETECTOR = "mask-glasses-yolov8m/weights/best.pt"
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3
N_INIT = 3
MAX_AGE = 30
DRAW_TRAILS = True
TRAIL_LEN = 20
# ------------------------------------------------


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]

def performOnVideo(track, frame, trails, mask_GLass_detector):
    track_id = track.track_id
    bbox = track.to_tlbr()  # tlbr = [min x,min y,max x,max y]
    x1, y1, x2, y2 = map(int, bbox)

    person_crop = frame[y1:y2, x1:x2].copy()

    mask_results = mask_GLass_detector(person_crop)
    res = mask_results[0]

    track.clear();
    # Update track mask/glasses status
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = res.names[cls_id]
        mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
        mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
        global_x1 = int(x1 + mx1)
        global_y1 = int(y1 + my1)
        global_x2 = int(x1 + mx2)
        global_y2 = int(y1 + my2)

        if "mask" in cls_name:
            track.setMask(cls_name,[ global_x1 ,global_y1 ,global_x2,global_y2 ])
        if "glass" in cls_name:
            track.setGlasses(cls_name, [ global_x1 ,global_y1 ,global_x2,global_y2 ])

    
    # Draw mask
    if track.mask_status == "mask":
        if track.mask_Cord is not None and len(track.mask_Cord) > 0:
            mx1, my1, mx2, my2 = map(int, track.mask_Cord)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            cv2.putText(frame, track.mask_status, (mx1, my1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Draw glasses
    if track.Glasses_status == "glass":
        if track.Glasses_Cord is not None and len(track.Glasses_Cord) > 0:
            gx1, gy1, gx2, gy2 = map(int, track.Glasses_Cord)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            cv2.putText(frame, track.Glasses_status, (gx1, gy1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw track bounding box + summary
    color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 29) % 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, track.GetSummary(), (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw center point & trail
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    if DRAW_TRAILS:
        if track_id not in trails:
            trails[track_id] = deque(maxlen=TRAIL_LEN)
        trails[track_id].appendleft((cx, cy))
        pts = list(trails[track_id])
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j-1], pts[j], color, 2)


def put_summary(frame, tracker):
    total_mask = 0
    total_glasses = 0
    total_people = 0
    wear_both = 0
    wear_none = 0

    for track in tracker.tracks:
        if track.mask_status == "mask" and track.Glasses_status == "glass":
            wear_both += 1
        
        if track.mask_status == "mask":
            total_mask += 1
        
        if track.Glasses_status == "glass":
            total_glasses += 1

        total_people += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  
    thickness = 2
    color = (0, 255, 0)

    y = 30
    line_gap = 20

    cv2.putText(frame, f"Total People: {total_people}", (10, y), font, font_scale, color, thickness)
    y += line_gap
    cv2.putText(frame, f"Wear Both (Mask + Glasses): {wear_both}", (10, y), font, font_scale, color, thickness)
    y += line_gap
    cv2.putText(frame, f"Wear Mask: {total_mask}", (10, y), font, font_scale, color, thickness)
    y += line_gap
    cv2.putText(frame, f"Wear Glasses: {total_glasses}", (10, y), font, font_scale, color, thickness)


def run_video(video_path, output_path=None):
    # Load YOLOv8 + Mask/Glasses detector
    yolo = YOLO(YOLO_MODEL)
    mask_GLass_detector = YOLO(MASK_GLASSES_DETECTOR)

    # Init DeepSORT
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=MAX_AGE, n_init=N_INIT)

    encoder = gdet.create_box_encoder(REID_WEIGHTS, batch_size=1)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trails = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLOv8 person detection
        results = yolo(frame, imgsz=640)
        res = results[0]

        person_boxes = []
        if hasattr(res, "boxes"):
            for box in res.boxes:
                cls = int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, "__getitem__") else float(box.conf)
                if conf < MIN_CONFIDENCE:
                    continue
                person_boxes.append([x1, y1, x2, y2, conf])

        detections = []
        if len(person_boxes) > 0:
            tlwhs = [xyxy_to_tlwh(p[:4]) for p in person_boxes]
            features = encoder(frame, tlwhs)
            for i, det in enumerate(person_boxes):
                detections.append(Detection(tlwhs[i], det[4], features[i]))

        tracker.predict()
        tracker.update(detections)

        # Draw tracks, mask & glasses (same as original code)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            performOnVideo(track, frame , trails , mask_GLass_detector)

            # bbox, crop, mask/glasses detection, drawing (as before)
            # ...
        put_summary(frame, tracker)
        if writer:
            writer.write(frame)
        cv2.imshow("Video Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(VIDEO_PATH, OUTPUT_PATH)



