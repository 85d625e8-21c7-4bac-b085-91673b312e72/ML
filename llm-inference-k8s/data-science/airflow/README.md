# AirFlow

Добавление репозитория AirFlow для helm
```bash
helm repo add airflow https://airflow.apache.org
```

Установка Helm Chart
```bash
helm install airflow airflow/airflow \
--debug \
--namespace airflow \
--create-namespace \
--set dags.gitSync.enabled=true \
--set dags.gitSync.repo=https://github.com/xLagerFeuer/006bc3974329b47cc3a9f93b7a2dcb1c.git \
--set dags.gitSync.branch=core \
--set dags.gitSync.subPath="/data-science/airflow/DAGs/" 
```

