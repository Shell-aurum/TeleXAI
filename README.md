# TeleXAI — Explainable Predictive Maintenance for 5G Networks

Predicting cell tower degradation is easy to demo and hard to trust.
Network engineers don't just want an alarm — they want to know **why**
a tower is being flagged before they act on it. TeleXAI is an
end-to-end pipeline that predicts 5G network degradation and explains
each prediction using SHAP and LIME, with a focus on measuring whether
those explanations are actually *correct*, not just plausible.

This project pairs with a from-scratch reproduction of
[*"Why Should I Trust You?": Explaining the Predictions of Any Classifier*](https://arxiv.org/abs/1602.04938)
(Ribeiro, Singh & Guestrin, 2016) — the paper that introduced LIME.
The reproduction lives in a companion repo/section (linked below) and
the theoretical grounding from it directly informs how explanations
are evaluated here.

## Why this is different from a typical "SHAP demo"

Most explainability projects show an explanation and stop, without
checking whether it's right. Because the telemetry here is synthetic
with **injected, ground-truth failure causes**, this project can go
one step further: for every predicted failure, compare what SHAP and
LIME say caused it against what actually caused it, and report a
faithfulness score for each method.

## Project structure

```
telexai/
├── data/
│   ├── raw/telemetry.csv       # synthetic 5G telemetry, 15 towers x 60 days
|   ├── featured/featured.csv
├── models/
├── src/
│   ├── generate_dataset.py     # synthetic data + failure injection
│   ├── features.py             # rolling/derived feature engineering
│   ├── train_model.py          # model training + evaluation
│   └── explain.py              # SHAP + LIME explanation + faithfulness scoring
├── notebooks/                  # exploratory analysis
├── dashboard/
│   └── app.py                  # Streamlit engineer-facing dashboard
├── reports/                    # write-ups, figures, results
├── requirements.txt
└── README.md
```

## Status / roadmap

- [x] Synthetic telemetry generator with 4 labeled failure modes
- [x] Feature engineering (rolling stats, rate-of-change)
- [x] Predictive models (LightGBM, RandomForest, XGBoost) + time-based evaluation
- [x] SHAP explanations (global + local)
- [x] LIME explanations
- [x] SHAP vs LIME faithfulness comparison against ground-truth cause
- [x] Streamlit dashboard
- [x] Write-up / report

## Dataset

See [`data/raw/README.md`](data/raw/README.md) for the full schema,
what each failure mode looks like, and — importantly — which columns
must be excluded from model features to avoid leaking the label.

Regenerate or resize the dataset:

```bash
python src/generate_dataset.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Author

Built as part of an applied XAI research portfolio, alongside a research endeavour

## License

MIT - see [LICENSE]