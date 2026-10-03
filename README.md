Predicting Stock Returns from Chart Images with a CNN

A simplified PyTorch implementation of **Jiang, Kelly & Xiu, "(Re-)Imag(in)ing Price Trends"** (*Journal of Finance*, 2023). Instead of hand-crafting technical indicators, a convolutional neural network looks at raw stock chart images and learns for itself which visual patterns predict future price movements.

**Task:** given a 20-day chart image of a stock, predict whether its return over the next 5 trading days will be positive (binary classification).

<p align="center">
  <img src="figures/sample_chart_1.png" width="30%">
  <img src="figures/sample_chart_2.png" width="30%">
  <img src="figures/sample_chart_3.png" width="30%">
</p>

## Highlights

- Trained on **~2.2 million chart images** of U.S. stocks covering **1993–2019**
- **Strict chronological train / validation / test split** (60/20/20 by date) to avoid look-ahead bias
- Lightweight CNN (**~72k parameters**) with batch normalization, dropout, weight decay, gradient clipping and early stopping
- Decision threshold tuned on the validation set, then applied unchanged to the test set
- Error analysis: confusion matrix, prediction confidence, and visual inspection of misclassified charts

## Data

Each sample is a 64 × 60 grayscale image encoding 20 trading days of price history: daily open/high/low/close bars, a 20-day moving average line, and volume bars along the bottom. Labels come from the accompanying feather files; the target is `Ret_5d > 0`.

The image dataset was released by the paper's authors and is **not included in this repository** because of its size. To run the notebook, download the `monthly_20d` image files and either place them in `data/monthly_20d/` or point the `DATA_DIR` environment variable at their location. The notebook expects files named like:

```
20d_month_has_vb_[20]_ma_{YEAR}_images.dat
20d_month_has_vb_[20]_ma_{YEAR}_labels_w_delay.feather
```

| Split      | Samples   | Period                    |
|------------|-----------|---------------------------|
| Train      | 1,458,026 | Jan 1993 – Feb 2009       |
| Validation | 363,309   | Mar 2009 – Jul 2014       |
| Test       | 369,389   | Aug 2014 – Dec 2019       |

Classes are close to balanced (about 50% "up" in every split).

## Model

```
Input (1 × 64 × 60)
 → Conv 3×3, 16  → BatchNorm → ReLU → MaxPool 2×2    # 32 × 30
 → Conv 3×3, 32  → BatchNorm → ReLU → MaxPool 2×2    # 16 × 15
 → Conv 3×3, 32  → BatchNorm → ReLU → MaxPool 2×2    #  8 × 7
 → Flatten → FC 32 → ReLU → Dropout(0.4) → FC 1 (logit)
```

Training setup: Adam (lr 3e-4, weight decay 1e-4), `BCEWithLogitsLoss`, batch size 32, up to 20 epochs with early stopping (patience 6) on validation F1.

This is a simplified version of the paper's architecture, which uses wider layers (64/128/256 channels), larger 5×3 kernels and LeakyReLU activations.

## Results (out-of-sample test set, 2014–2019)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 52.48% |
| Precision | 0.528  |
| Recall    | 0.588  |
| F1 (Up)   | 0.556  |

<p align="center">
  <img src="figures/confusion_matrix.png" width="45%">
  <img src="figures/threshold_tuning.png" width="50%">
</p>

**How to read these numbers.** Short-horizon stock returns are extremely noisy and close to unpredictable, so accuracy only slightly above 50% is the expected outcome for this kind of problem. The original paper also reports accuracies just a few points above a coin flip; its contribution is showing that this small but consistent edge, applied across thousands of stocks, translates into economically meaningful portfolio returns. This model reaches a 52.5% hit rate on more than 369k unseen samples from a later time period than anything it was trained on.

### Training dynamics

![Training curves](figures/training_curves.png)

Validation loss is lowest after the first epoch and rises afterwards while training accuracy keeps improving, a sign that the model quickly starts fitting noise in the training period. Early stopping restores the best checkpoint. This mirrors a central difficulty of financial machine learning: the signal-to-noise ratio is very low, and patterns in one market regime do not fully carry over to the next.

### Error analysis

![Misclassified examples](figures/misclassified_examples.png)

The model's average confidence is only marginally higher on correct predictions than on incorrect ones, which again reflects how weak the predictive signal is at the individual-stock level.

## Project structure

```
├── stock_cnn.ipynb      # Full pipeline: loading, splitting, training, evaluation
├── figures/             # Plots used in this README
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
# put the data in data/monthly_20d/ (or set DATA_DIR), then:
jupyter notebook stock_cnn.ipynb
```

A CUDA GPU is strongly recommended. Loading all years into memory requires roughly 9 GB of RAM for the images alone.

## Possible extensions

- Implement the paper's full architecture and compare against this simplified version
- Ensemble several independently trained models, as the paper does
- Build long-short decile portfolios from the predicted probabilities and evaluate Sharpe ratios
- Try other horizons (5-day or 60-day images, 20-day or 60-day returns)
- Use Grad-CAM to visualise which parts of a chart drive the predictions

## Reference

Jiang, J., Kelly, B., & Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. *The Journal of Finance*, 78(6), 3193–3249.
