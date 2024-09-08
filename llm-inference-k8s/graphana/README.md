# Graphana panel

## Deploy

```bash
kubectl create namespace monitoring
```

```bash
kubectl apply -f 00-grafana-config.yaml

```

```bash
kubectl apply -f 01-grafana-pvc.yaml
```

```bash
kubectl apply -f 02-grafana-deployment.yaml
```

```bash
kubectl apply -f 03-grafana-service.yaml
```

Получение статуса запуска контейнера:

```bash
kubectl get pods -n monitoring
```

После применения всех файлов, настройте порт форвардинг, чтобы получить доступ к Grafana через локальный браузер:

```bash
kubectl port-forward --namespace monitoring service/grafana 3000:80
```

Теперь Grafana будет доступна по адресу http://localhost:3000.