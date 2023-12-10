import torch
from torch import nn


def load_pretrained_yolov5(model_name: str = "yolov5s") -> nn.Module:
    """Load a pretrained YOLOv5 model.

    Args:
    model_name (str): Name of the YOLOv5 model to be loaded. Default is 'yolov5s'.
        Available models are 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'.
        The 's', 'm', 'l', and 'x' variants represent small, medium, large, and
        extra-large models respectively.

    Returns:
    nn.Module: A PyTorch YOLOv5 model loaded with pretrained weights.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # URL for the YOLOv5 GitHub repository
    github_repo_url = "ultralytics/yolov5"

    # Load the pretrained model from the official YOLOv5 repository
    model = torch.hub.load(github_repo_url, model_name, pretrained=True)  # type: ignore

    return model.eval().to(device)
