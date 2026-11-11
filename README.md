# BloodMNIST Cell Classifier

An educational image-classification project that trains a small convolutional
neural network (CNN) to classify blood-cell images into the eight BloodMNIST
classes. It uses MedMNIST's official train, validation, and test splits.

> **Notice:** This project is for learning and experimentation only. It is not
> a medical device and must not be used to diagnose or guide treatment.

## Project contents

| File or directory | Purpose |
|---|---|
| `train.py` | Defines the CNN, trains it, selects the best checkpoint using validation accuracy, and reports test metrics. |
| `inference.py` | Loads a saved checkpoint and predicts a cell type from an image file or a test-split sample. |
| `requirements.txt` | Python package dependencies. |
| `data/` | Downloaded BloodMNIST data; created automatically. |
| `models/` | Saved model checkpoints; created automatically. |
| `env/` | Local Python virtual environment; create during setup. |

The `data/`, `models/`, and `env/` directories are excluded from Git.

## Requirements
- Python 3.11 or newer
- Internet access on the first run to install dependencies and download
  BloodMNIST

PyTorch uses CUDA automatically when available; otherwise, the project runs on
the CPU.

## Setup

Run these commands in PowerShell from the project directory:

```powershell
cd D:\projects\TissueIdentifier
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, call its Python executable
directly:

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

With the environment activated:

```powershell
python train.py
```

Or without activating it:

```powershell
.\env\Scripts\python.exe train.py
```

By default, training runs for 15 epochs with batch size 128 and learning rate
`0.001`. The script downloads the dataset into `data/` if needed. At each
epoch, it evaluates the validation split and saves the checkpoint whenever
validation accuracy improves. The checkpoint is written to
`models/best_model.pt`.

After training, the script restores the best validation checkpoint and reports
test accuracy, per-class precision/recall/F1, and a confusion matrix.
BloodMNIST's predefined split contains 11,959 training images, 1,712 validation
images, and 3,421 test images (approximately 70% / 10% / 20%).

Optional training arguments:

```powershell
python train.py --epochs 25 --batch-size 128 --learning-rate 0.001
```

Other supported options include `--data-dir`, `--output`, and `--seed`. Use
`python train.py --help` to see all options. The default random seed is `42`.

The held-out test split is for final evaluation. Avoid changing model settings
based on repeated test-set results; use validation metrics for model selection.

## Run inference

### Predict from an image file

After training, pass a PNG, JPEG, or other Pillow-readable image:

```powershell
python inference.py "D:\path\to\cell-image.png"
```

The script converts the image to RGB, resizes it to 28×28 pixels, and prints
the predicted cell type and probabilities for all eight classes. To load a
checkpoint from another location:

```powershell
python inference.py "D:\path\to\cell-image.png" --checkpoint "D:\path\to\model.pt"
```

### Predict from BloodMNIST's held-out test split

BloodMNIST is stored as a compressed dataset rather than individual image
files. To classify one of its test images, use its zero-based index (0 through
3420):

```powershell
python inference.py --test-sample 0
```

This prints the predicted class, the known test label, whether they match, and
the predicted probabilities. This command uses the image only for inference
and comparison; it does not train or update the model. The same test set is
also used for the final metrics printed by `train.py`.

Optional inference arguments include `--checkpoint` and `--data-dir`. Use
`python inference.py --help` to see all options.

## Model overview

The small CNN uses three convolutional blocks with 32, 64, and 128 feature
channels. The first two blocks include max pooling, reducing a 28×28 image to
7×7 spatial feature maps. Adaptive average pooling summarizes those maps,
then dropout and a linear layer produce eight class logits. Training uses
cross-entropy loss and the AdamW optimizer.

## Dependencies

Install the packages listed in [`requirements.txt`](requirements.txt). The
main dependencies are PyTorch, MedMNIST, NumPy, scikit-learn, and Pillow.
