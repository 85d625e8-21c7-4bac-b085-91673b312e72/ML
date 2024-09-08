import mlflow
import mlflow.pyfunc
from pathlib import Path


class YOLOv8PoseModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO
        model_path = context.artifacts["yolo_model"]
        self.model = YOLO(model_path)

    def predict(self, context, model_input):
        results = self.model(model_input)
        return results.pandas().xyxy[0]  # Выводим результаты в формате pandas DataFrame


def log_yolo_model(model_path, model_name="yolov8_pose_model"):
    # Путь к модели YOLOv8
    model_path = Path(model_path)

    # Начало логирования модели
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=YOLOv8PoseModel(),
            artifacts={"yolo_model": str(model_path)},
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'pip',
                    {
                        'pip': [
                            'mlflow',
                            'ultralytics'
                        ],
                    },
                ],
                'name': 'yolo_env'
            }
        )
        print(f"Model logged in run {run.info.run_id}")



model_path = "numbers_best"
log_yolo_model(model_path)
