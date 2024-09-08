# Automatisation Service using n8n

![example.png](__assets__%2Fexample.png)

## Install

```
kubectl apply -k .
```

You should see in the `default` namespace defined for kubernetes all the new objects.
The corresponding service is called `n8n`.

If you want to use it in a different namespace then just add "-n &lt;namespace&gt;" as argument to kubectl call.

You need to use a proxy in front of it (like nginx) to redirect to the exposed port on k8s node.
To find the mapped ports for port 5678 just check the "Internal Endpoints" of the service "n8n"
or run `kubectl get svc` from the shell of the server running the cluster.


