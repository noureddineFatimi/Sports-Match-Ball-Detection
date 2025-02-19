from ultralytics import YOLO

model = YOLO("best.pt")
video_image_path = "testing/twitch.jpg"
results = model.predict(video_image_path, save=True, conf=0.5)