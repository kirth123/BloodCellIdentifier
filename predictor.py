"""Load a checkpoint once and classify PIL images without dataset dependencies."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import SmallCNN


class Predictor:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Run python train.py first, or set BLOODCELL_CHECKPOINT to a trained checkpoint."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.class_names = checkpoint["class_names"]
        self.model = SmallCNN(len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((28, 28), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return torch.softmax(self.model(tensor), dim=1)[0].cpu()
