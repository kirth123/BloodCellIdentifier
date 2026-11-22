"""Local Gradio interface: run with python app.py."""

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from predictor import Predictor


def create_app(checkpoint_path: Path | None = None) -> gr.Interface:
    checkpoint_path = checkpoint_path or Path(
        os.environ.get(
            "BLOODCELL_CHECKPOINT",
            str(Path(__file__).resolve().parent / "models" / "best_model.pt"),
        )
    )
    predictor = None
    startup_error = None
    try:
        predictor = Predictor(checkpoint_path)
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        startup_error = str(error)

    def classify(image: Image.Image | None) -> dict[str, float]:
        if image is None:
            raise gr.Error("Upload a cropped image of a single blood cell first.")
        if predictor is None:
            raise gr.Error(startup_error or "The model could not be loaded.")
        probabilities = predictor.predict(image)
        return dict(zip(predictor.class_names, probabilities.tolist()))

    status = (
        "**Model ready.** Upload an image and select Classify."
        if predictor is not None
        else "**Model unavailable.** Train a model with `python train.py`, "
        "or set `BLOODCELL_CHECKPOINT` to your saved checkpoint, then restart this app."
    )
    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", sources=["upload"], label="Single-cell image"),
        outputs=gr.Label(num_top_classes=8, label="Predicted cell type and probabilities"),
        title="Blood Cell Classifier",
        description=(
            "Upload a cropped microscope image of a single blood cell. "
            "The model classifies it into one of eight BloodMNIST cell types. "
            "Full blood-smear images are not supported.\n\n" + status
        ),
        article=(
            "For learning and experimentation only; not for diagnosis or treatment. "
            "Images are resized to 28 × 28 pixels. The model always chooses among its "
            "known cell types, even for unrelated images; probabilities do not establish correctness."
        ),
        submit_btn="Classify",
        flagging_mode="never",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(share=False)
