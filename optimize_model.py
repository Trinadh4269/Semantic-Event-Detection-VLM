import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

quantized_model = torch.quantization.quantize_dynamic(
    model.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "optimized_model.pt")
print("Optimized model saved successfully")
