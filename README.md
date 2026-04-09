# Deep Neural Networks with Ordinal Loss for Medical Applications

Code for the paper:

> **Deep Neural Networks with Ordinal Loss for Medical Applications**  
> Tal Dvora, Rotem Haba, Gonen Singer  
> Bar-Ilan University, Ramat Gan, Israel

## Overview

This repository provides the experimental framework for evaluating the proposed **Ordinal Cross-Entropy (OCE)** loss against existing ordinal and cost-sensitive loss functions on the APTOS 2019 Blindness Detection dataset. The task is 5-class ordinal classification of diabetic retinopathy (DR) severity (grades 0–4).

OCE extends standard cross-entropy to incorporate a cost matrix that encodes both the distance between ordinal classes and the direction of misclassification (overestimation vs. underestimation), reflecting clinically meaningful risk.

## Project Structure

```
compare-models/
├── main.py              # Entry point: runs training and evaluation loops
├── models_option.py     # Model definitions, training, and evaluation logic
├── loss_option.py       # Loss function implementations
├── preprocess.py        # Data loading, preprocessing, and fold creation
├── distributions.py     # Beta, Poisson, Binomial, Exponential distributions (for regularized baselines)
├── config.py            # Penalty/weight matrix configurations
├── t-test.py            # Statistical significance testing on results
└── data/
    └── APTOS 2019 Blindness Detection/
        └── train/       # Training images + fold CSVs (auto-generated on first run)
```

## Loss Functions

| Key | Description | Paper notation |
|-----|-------------|---------------|
| `ordinal_cross_entropy_loss` | **Proposed OCE** — cross-entropy with ordinal cost matrix | OCE |
| `cross_entropy_loss` | Standard categorical cross-entropy | CE |
| `categorical_ce_beta_regularized` | CE with unimodal Beta label regularization | CE-β |
| `categorical_ce_poisson_regularized` | CE with Poisson label regularization | CE-P |
| `categorical_ce_binomial_regularized` | CE with Binomial label regularization | CE-B |
| `categorical_ce_exponential_regularized` | CE with Exponential label regularization | CE-E |
| `ordinal_loss` | Distance-aware ordinal penalty loss (Chen et al., 2019) | OL |

## Architectures

All models use ImageNet pretrained weights with frozen base layers and a custom 5-class output head (leaky-ReLU activations, no final softmax — handled inside the loss).

| Model | Key |
|-------|-----|
| DenseNet-121 | `densenet121` |
| InceptionV3 | `inceptionv3` |
| VGG-19 | `vgg19` |
| ResNet-50/101/152 | `resnet50`, `resnet101`, `resnet152` |
| MobileNet | `mobilenet` |
| AlexNet (custom) | `alexnet` |

## Experimental Setup

The paper evaluates OCE against all baselines on three architectures (DenseNet121, InceptionV3, VGG19) using 5-fold cross-validation, 25 training epochs, Adam optimizer (lr=1e-3), batch size 32, under both **symmetric** and **asymmetric** cost matrices.

To reproduce the paper's main experiments, configure `main.py`:

```python
set_name        = 'APTOS 2019 Blindness Detection'
data_op         = 'train'
loss_op_list    = ['ordinal_cross_entropy_loss']   # swap to compare baselines
model_name_list = ['densenet121', 'inceptionv3', 'vgg19']
callback_types  = ['reduce_lr']
epochs          = 25
n_splits        = 5
smote_op_list   = [False]
```

## Running

```bash
python main.py
```

Fold CSVs are created automatically on first run under `data/<set_name>/train/` and reused on subsequent runs.

## Outputs

- **Per-fold results**: `results/<set_name>/fold{i}_{loss}_{callback}_{model}_{smote}_results.xlsx`
  - Sheets: `Metrics`, `Confusion_Matrix_Train`, `Confusion_Matrix_Val`
- **Training curves**: `graphs/<set_name>/fold{i}_{model}_{loss}_{callback}_{smote}_metrics.png`
- **Summary CSV**: `results/<set_name>/<data_op>_summary_metrics.csv`

## Evaluation Metrics

The primary criterion is **mean misclassification cost** under the task-specific ordinal cost matrix (symmetric or asymmetric). Complementary metrics reported:

- Accuracy, AUC (OvR)
- MAE, Quadratic Weighted Kappa (QWK)
- Per-class precision and recall
- Confusion matrices (train / validation)

## Dependencies

- Python 3.9
- TensorFlow / Keras
- scikit-learn
- imbalanced-learn
- pandas, numpy, matplotlib, scipy, xlsxwriter

## Citation

If you use this code, please cite:

```
Tal Dvora, Rotem Haba, Gonen Singer.
"Deep Neural Networks with Ordinal Loss for Medical Applications."
```
