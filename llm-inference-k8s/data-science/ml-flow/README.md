# ML FLow

## k8s pipline

## Deploy

```bash
kubectl apply -f 00-mlflow-deployment.yaml
```

```bash
kubectl apply -f mlflow-service.yaml
```

```bash
kubectl get all -n mlflow
```

Если все работает правильно, вы сможете получить доступ к MLflow UI через внешний IP-адрес вашего сервиса LoadBalancer. Если внешний IP-адрес не назначен автоматически, вы можете использовать kubectl port-forward для временного перенаправления порта:

```bash
kubectl port-forward svc/mlflow 5000:5000 -n mlflow
```

После этого откройте браузер и перейдите по адресу http://localhost:5000, чтобы получить доступ к интерфейсу MLflow.

### If you want open port in your OS

```bash
kubectl port-forward --address 0.0.0.0 svc/mlflow 5000:5000 -n mlflow
```

## Potantial problems
Проверка логов: Если проблема не решена, проверьте логи для получения дополнительной информации о причине ошибки:

```bash
kubectl describe pod <pod_name> -n mlflow
kubectl logs <pod_name> -n mlflow
```

