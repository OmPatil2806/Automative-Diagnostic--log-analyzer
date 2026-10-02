# Architecture

```
            ┌──────────────┐     ┌──────────────────┐
            │  VED logs    │     │ Synthetic logs   │
            │ (data/raw)   │     │ (data/synthetic) │
            └──────┬───────┘     └────────┬─────────┘
                   └──────────┬───────────┘
                              ▼
                   ingestion/  load + clean
                              ▼
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
   dtc/ decode codes   features/ build features   │
          │                   ▼                    │
          │          detection/ anomaly model      │
          └───────────────────┼────────────────────┘
                              ▼
            analysis/ correlation + health score
                              ▼
                 reporting/ diagnosis report
                              ▼
                  dashboard/ (Streamlit)
```

| Module | Responsibility |
|---|---|
| `ingestion` | Read raw and synthetic logs, clean them into one standard format |
| `dtc` | Translate fault codes into meaning, severity and likely causes |
| `synthetic` | Generate logs with known, labeled faults |
| `features` | Turn raw signals into ML features |
| `detection` | Find abnormal sensor behavior |
| `analysis` | Link anomalies to faults, compute the health score |
| `reporting` | Produce the final diagnosis report |
| `pipeline.py` | Run all steps end to end |
