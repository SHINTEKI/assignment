"""
### Model Versioning
I would use tools like Weights & Biases (WandB) to version and track models in production. Each model version would be logged with associated experiment metadata, including train setting, hyperparameters, dataset version (data can be tracked using DVC), performance metrics, and saved artifacts (such as model weights, model.ckpt). This makes it easy to reproduce results, compare experiments side by side, and roll back to a previous model if needed.

I can also add manual tag to indicate the version of model deployed for faster search.


### Monitoring Metrics
Key metrics include:
- API: custom logs, latency, error rate, request volume to monitor software health and failure patterns.
- Model: input feature and prediction distribution to monitor data and target drift.
"""