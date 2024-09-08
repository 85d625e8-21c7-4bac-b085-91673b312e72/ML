# Архитектура

```mermaid
graph TD;
    A[User Requests] -->|API Gateway| B[Load Balancer];
    B --> C[Ingress Controller];
    C --> D[Kubernetes Cluster];
    
    subgraph LLM Inference Cluster
        D1[CPU Preprocessing Node] --> D2[GPU Inference Node];
        D2 -->|Model Cache| D3[Redis];
    end
    
    D --> E[Model Storage (S3)];
    
    subgraph Monitoring and Observability
        F1[Prometheus];
        F2[Grafana];
        F3[Elasticsearch];
        F4[Jaeger];
    end
    
    D --> F1;
    D --> F2;
    D --> F3;
    D --> F4;
    
    F1 --> F2;
    F3 --> F4;
```
