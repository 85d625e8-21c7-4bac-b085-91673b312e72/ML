# KrakenD AGW

## Installation:

```bash
kubectl create -f deployment-definition.yaml
```

```bash
kubectl create -f service-definition.yaml
```

Check the health:

```bash
kubectl get svc
```

Port Forwarting:

```bash
kubectl port-forward svc/krakend-service 8080:80
```

## Alternative using Helm:
```bash
helm repo add equinixmetal https://helm.equinixmetal.com
```


```bash
helm install krakend equinixmetal/krakend
```

[docs](https://github.com/equinixmetal-helm/krakend)

## Origin on:

[Official KrakenD k8s docs](https://www.krakend.io/docs/deploying/kubernetes/)
