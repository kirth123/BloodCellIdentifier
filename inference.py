import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a blood-cell image with a trained BloodMNIST model."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to an image file (use this or --test-sample)",
    )
    parser.add_argument(
        "--test-sample",
        type=int,
        help="Classify a sample by index from BloodMNIST's official test split",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="MedMNIST data directory (default: data)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_model.pt"),
        help="Model checkpoint (default: models/best_model.pt)",
    )
    args = parser.parse_args()

    if (args.image is None) == (args.test_sample is None):
        parser.error("provide either an image path or --test-sample, but not both")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {args.checkpoint}")

    predictor = Predictor(args.checkpoint)
    class_names = predictor.class_names

    actual_index: int | None = None
    if args.test_sample is not None:
        import medmnist
        from medmnist import INFO

        dataset_info = INFO["bloodmnist"]
        dataset_class = getattr(medmnist, dataset_info["python_class"])
        dataset = dataset_class(
            split="test",
            root=str(args.data_dir),
            download=True,
        )
        if not 0 <= args.test_sample < len(dataset):
            parser.error(
                f"--test-sample must be between 0 and {len(dataset) - 1}"
            )
        image, label = dataset[args.test_sample]
        actual_index = int(np.asarray(label).reshape(-1)[0])
    else:
        if not args.image.is_file():
            parser.error(f"image file does not exist: {args.image}")
        with Image.open(args.image) as opened_image:
            image = opened_image.copy()

    probabilities = predictor.predict(image)

    predicted_index = int(probabilities.argmax())
    print(f"Predicted cell type: {class_names[predicted_index]}")
    if actual_index is not None:
        actual_name = class_names[actual_index]
        print(f"Actual test label: {actual_name}")
        print(f"Correct: {predicted_index == actual_index}")
    print("Class probabilities:")
    for index in probabilities.argsort(descending=True).tolist():
        print(f"  {class_names[index]}: {probabilities[index].item():.4f}")


if __name__ == "__main__":
    main()
