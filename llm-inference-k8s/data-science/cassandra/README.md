# Cassandra k8s manifest

```bash
kubectl apply -f cassandra-service.yaml
```

Провека cassandra pods:

```bash
kubectl get svc cassandra
```

Создание кольца cassandra:

```bash
kubectl apply -f cassandra-statefulset.yaml
```