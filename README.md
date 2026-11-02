# RCLiFT

Resistance–Capacitance-inspired Liquid-state Fusion Transformer for electricity-load forecasting.

The model predicts 24 hourly loads from 168 observed hours. The released configuration has 115,288 trainable parameters.

## Files

- `rclift.py`: standalone PyTorch model.
- `model_config.json`: architecture settings and checkpoint SHA-256 hashes.
- `weights/gefcom/seed{42,43,44}.pt`: GEFCom2014 weights.
- `weights/pjme/seed{42,43,44}.pt`: PJME weights.

Each checkpoint is an independently trained model state dictionary.

## Load a checkpoint

```bash
pip install -r requirements.txt
```

Run from the repository directory:

```python
import json
from pathlib import Path

import torch
from rclift import RCLiFT

config = json.loads(Path("model_config.json").read_text(encoding="utf-8"))
model = RCLiFT(**config["constructor_kwargs"])
weights = torch.load("weights/gefcom/seed42.pt", map_location="cpu", weights_only=True)
model.load_state_dict(weights, strict=True)
model.eval()

# x: preprocessed float32 tensor with shape [batch, 168, 22]
with torch.inference_mode():
    prediction = model(x)  # [batch, 24]
```

Inputs must use the feature order and normalisation used to train the selected dataset's checkpoint. The first two channels are load and temperature; the remaining channels contain the original weather and calendar features. Historical load and temperature differences are computed inside the model. Predictions use the input load's normalised scale; recover MW using the corresponding dataset preprocessing scale and minimum: `prediction_mw = prediction * scale + minimum`.
