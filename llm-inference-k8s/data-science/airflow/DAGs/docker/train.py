from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics import settings


# Инициализация Roboflow
rf = Roboflow(api_key="KdzwPt1kMDiWW0rj6uv7")
project = rf.workspace("idk-ocra0").project("bfhdofg")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded and unpacked to:", dataset.location)

settings.update({
    'mlflow': True,
    'clearml': False,
    'comet': False,
    'dvc': False,
    'hub': False,
    'neptune': False,
    'raytune': False,
    'tensorboard': False,
    'wandb': False
})

# Инициализация и тренировка модели YOLOv8
model = YOLO("yolov8n-pose.yaml")
model.train(data=dataset.location, epochs=20)

# Оценка модели
results = model.val()
print("Validation results:", results)

# Использование модели для предсказаний
predictions = model.predict("./valid/image.jpg")
print("Predictions:", predictions)
