from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import supervision as sv


class BallAnnotator:
    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        xy = detections.xyxy[:, :2].astype(int)  # Utiliser les coordonnées (x, y) de la détection
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.xyxy[:, :2]
        self.buffer.append(xy)

        if len(detections.xyxy) == 0:
            return detections

        centroid = np.mean(np.vstack(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return sv.Detections(xyxy=np.array([detections.xyxy[index]]))  # Utiliser `xyxy`


# Charger le modèle YOLO
model = YOLO("best.pt")

# Chemin de la vidéo
video_path = "testing/Video4.mp4"
output_path = "output/Video4.mp4"

# Lire la vidéo
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialiser les classes
ball_annotator = BallAnnotator(radius=15, buffer_size=10)
ball_tracker = BallTracker(buffer_size=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection avec YOLO
    results = model.predict(frame, save=False, conf=0.5)

    # Extraire les coordonnées des boîtes englobantes
    boxes = results[0].boxes
    if len(boxes) > 0:
        xyxy = np.array([box.xyxy[0].tolist() for box in boxes])  # Convertir en format (x1, y1, x2, y2)
        detections = sv.Detections(xyxy=xyxy)
    else:
        detections = sv.Detections(xyxy=np.empty((0, 4)))  # Aucune détection

    # Mise à jour du tracker
    tracked_detections = ball_tracker.update(detections)

    # Annotation de la trajectoire
    annotated_frame = ball_annotator.annotate(frame, tracked_detections)

    # Afficher et sauvegarder la vidéo
    cv2.imshow('Trajectoire avec cercles', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
