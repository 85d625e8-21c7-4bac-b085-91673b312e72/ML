from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime
from airflow import DAG
import os

os.system("docker build -t rs-update-dataset:latest -f docker/dataset.Dockerfile .")

with DAG(
    dag_id="update-dataset",
    schedule=None,
    start_date=datetime.now(),
    catchup=False,
    tags=["example"],
) as dag:
    airflow_with_kubernetes = KubernetesPodOperator(
        name="kubernetes_operator",
        image="rs-update-dataset:latest",
        cmds=["python"],
        arguments=["dataset_builder.py"],
        task_id="run-pod-update-dataset",
    )

airflow_with_kubernetes.dry_run()