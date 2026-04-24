# MalariaVision: Automated Malaria Detection Using Deep Learning

> **AIMS RIC Doctoral Training Programme — Kigali, Rwanda**  
> Martha Kachweka · Hajara Kandeh · Hilary Chaleu · Festa Ndubuogaranya · Jeannette Nyirahakizimana

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-malariaVision-blue)](https://huggingface.co/spaces/Agonza/malariaVision)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)

---

## Overview

MalariaVision is a deep learning pipeline for automated binary classification of blood smear cell images as **Parasitized** (malaria-infected) or **Uninfected** (healthy). The system evaluates six architectures — a custom baseline CNN, five ImageNet pre-trained models, and YOLOv8-n — on the NIH Malaria Cell Image Dataset. The best-performing model is deployed as a publicly accessible web application with integrated Grad-CAM explainability.

---

## Repository Structure

```
MalariaVision/
│
├── 3_ImageNetModels.ipynb       # Baseline CNN, MobileNetV2, DenseNet121, VGG16
├── Resnet50_group2.ipynb        # ResNet-50 transfer learning pipeline
├── efficientnet.ipynb           # EfficientNet-B0 pipeline
├── yoloV8-n_Model.ipynb         # YOLOv8-n training, evaluation & GradCAM
│
├── app.py                       # Gradio deployment application
├── requirements.txt             # Dependencies for HuggingFace deployment
├── best.pt                      # Trained YOLOv8-n weights
└── README.md
```

---

## Dataset

**NIH Malaria Cell Image Dataset**  
- Source: [NIH / Kaggle](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip)
- Total images: **27,558** (perfectly balanced)
- Parasitized: 13,779 | Uninfected: 13,779
- Resolution: 130×130 pixels (resized to 128×128 for training)

### Augmentation

Horizontal flipping was applied to each training image, expanding the dataset to **55,116 images** and improving model invariance to slide orientation variability.

### Train / Val / Test Split

| Split | Ratio | Images |
|-------|-------|--------|
| Train | 70%   | 38,582 |
| Val   | 15%   | 8,267  |
| Test  | 15%   | 8,267  |

---

## Models Evaluated

| Model | Framework | Parameters | Input Size | Epochs |
|-------|-----------|-----------|------------|--------|
| Baseline CNN | TensorFlow/Keras | ~200K | 128×128 | 20 |
| VGG16 | TensorFlow/Keras | 14.9M | 224×224 | 20 |
| MobileNetV2 | TensorFlow/Keras | 2.4M | 224×224 | 20 |
| DenseNet121 | TensorFlow/Keras | 7.0M | 224×224 | 20 |
| EfficientNet-B0 | TensorFlow/Keras | 5.3M | 224×224 | 20 |
| ResNet-50 | PyTorch | 25.6M | 224×224 | 20 |
| **YOLOv8-n** | **PyTorch / Ultralytics** | **3.2M** | **128×128** | **50** |

All ImageNet models followed a two-phase training strategy: base layers frozen with only the classification head trained, followed by fine-tuning at a reduced learning rate of 1×10⁻⁵.

---

## Results

| Model | Class | Precision | Recall | F1-score |
|-------|-------|-----------|--------|----------|
| VGG16 | Parasitized | 0.98 | 0.96 | 0.97 |
| | Uninfected | 0.96 | 0.98 | 0.97 |
| MobileNetV2 | Parasitized | 0.97 | 0.96 | 0.97 |
| | Uninfected | 0.96 | 0.97 | 0.97 |
| ResNet-50 | Parasitized | 0.98 | 0.98 | 0.98 |
| | Uninfected | 0.98 | 0.98 | 0.98 |
| **YOLOv8-n** | **Parasitized** | **0.98** | **0.97** | **0.98** |
| | **Uninfected** | **0.97** | **0.98** | **0.98** |

**YOLOv8-n** achieves the highest overall F1-score of **0.98** for both classes while requiring only ~3.2M parameters — making it the optimal choice for resource-constrained deployment.

---

## Explainability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to YOLOv8-n to visualise which regions of the cell image influenced the model's prediction. Heatmaps consistently highlight the dark-stained *Plasmodium* parasite regions in parasitized cells, confirming the model attends to biologically meaningful features rather than background artefacts.

---

## Deployment

The YOLOv8-n model is deployed as an interactive web application on **Hugging Face Spaces**:

🔗 **[huggingface.co/spaces/Agonza/malariaVision](https://huggingface.co/spaces/Agonza/malariaVision)**

The application accepts a blood smear cell image as input and returns:
- Predicted class (Parasitized / Uninfected)
- Confidence score
- Grad-CAM overlay highlighting the regions that influenced the prediction

---

## Installation & Local Setup

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Agonza/malariaVision
cd malariaVision

# Install dependencies
pip install -r requirements.txt

# Run the app locally
python app.py
```

### Requirements

```
ultralytics
gradio
torch
torchvision
opencv-python-headless
Pillow
numpy
```

---

## Running the Notebooks

All notebooks are designed to run on **Kaggle** with GPU acceleration.

| Notebook | Description |
|----------|-------------|
| `3_ImageNetModels.ipynb` | Data augmentation, baseline CNN, MobileNetV2, DenseNet121, VGG16 |
| `Resnet50_group2.ipynb` | ResNet-50 with custom PyTorch Dataset and DataLoader |
| `efficientnet.ipynb` | EfficientNet-B0 pipeline |
| `yoloV8-n_Model.ipynb` | YOLOv8-n training, evaluation, GradCAM, saliency maps |

**Dataset path on Kaggle:**
```
/kaggle/input/datasets/marthakachweka/cell-data/cell_images
```

---

## Key Dependencies

```python
# Deep learning
torch >= 2.0
tensorflow >= 2.12
ultralytics >= 8.0

# Data & visualisation
numpy
pandas
matplotlib
opencv-python-headless
Pillow
scikit-learn

# Deployment
gradio
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{malariaVision2025,
  title     = {MalariaVision: Building a Deep Learning System 
               for Automated Malaria Detection},
  author    = {Kachweka, Martha and Kandeh, Hajara and Chaleu, Hilary 
               and Ndubuogaranya, Festa and Nyirahakizimana, Jeannette},
  year      = {2025},
  institution = {AIMS RIC Doctoral Training Programme, Kigali, Rwanda}
}
```

---

## Acknowledgements

- NIH / Rajaraman et al. for the Malaria Cell Image Dataset
- Ultralytics for the YOLOv8 framework
- Hugging Face for free model hosting
- AIMS RIC Doctoral Training Programme

---

## License

This project is for academic and research purposes.  
© 2026 AIMS RIC Doctoral Training Programme, Kigali, Rwanda.
