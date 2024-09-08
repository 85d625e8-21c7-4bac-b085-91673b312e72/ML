# Prometheus

```bash
kubectl apply -f 00-prometheus-config.yaml
```

```bash
kubectl apply -f 01-prometheus-deployment.yaml
```

```bash
kubectl apply -f 02-prometheus-service.yaml
```

```bash
kubectl apply -f 03-myapp-servicemonitor.yaml
```

## Настройка Grafana для использования Prometheus

Если Grafana уже установлена, добавьте источник данных Prometheus в Grafana, используя URL вашего Prometheus сервиса (например, http://prometheus.monitoring.svc.cluster.local:9090).

