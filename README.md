# Customer Churn Prediction

Predicting which telecom customers are likely to churn, and identifying the drivers behind
it, using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, 21 features).

## Project Goal

Go beyond "who churns" to answer **why**, and translate that into a retention strategy:
clean and audit the data, explore churn patterns across contract type / pricing / service
usage, engineer features that capture service adoption, then train and compare two
classification models to rank the strongest churn drivers.

## Results

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Logistic Regression | 80.7% | 0.66 | 0.56 | 0.60 |
| Random Forest | 79.2% | 0.64 | 0.51 | 0.56 |

Both models land in the same range and both under-recall the minority (churn) class —
expected on an imbalanced target (~27% churn rate) without resampling. See the
[**Next steps**](#next-steps) below for how to push recall further.

**Top churn drivers** (Random Forest feature importance): `TotalCharges`,
`CostPerService` *(engineered)*, `tenure`, `MonthlyCharges`, `PaymentMethod`. Billing and
tenure dominate the ranking — with a caveat discussed in the notebook: tree-based
importance is biased toward continuous features, so categorical signals like `Contract`
and `InternetService` (both clearly linked to churn in the EDA) likely matter more than
their raw importance score suggests.

## Key Findings

- **Month-to-month contracts churn far more** than one/two-year contracts — no commitment
  makes leaving frictionless.
- **Price is a factor**: churned customers pay higher `MonthlyCharges` on median, and
  Fiber optic — the most expensive internet tier — also has the highest churn rate.
- **Service adoption retains**: churn falls steadily from 2 add-on services upward, down
  to 5.3% for customers with all 6 (vs. 45.8% for customers with just 1). Tech Support
  subscribers in particular churn much less.
- **Tenure is the strongest continuous signal**: the longer someone stays, the less likely
  they are to leave — the model's clearest, most intuitive finding.

**Business recommendation:** prioritize retention outreach (discounts, contract upgrade
offers, free Tech Support trials) at customers in their first months on a month-to-month,
Fiber optic contract with few add-on services — the segment where these risk factors stack.

## Repo Structure

```
customer-churn-analysis/
├── churn_prediction.ipynb   # Full analysis: cleaning → EDA → feature engineering → modeling → findings
├── data/
│   └── telco_customer_churn.csv
├── requirements.txt
└── README.md
```

## Methodology

1. **Data cleaning** — `TotalCharges` is coerced to numeric; the 11 rows left blank
   (all brand-new customers with `tenure == 0`) are imputed with `MonthlyCharges` rather
   than 0, since that's the one full month they've actually been billed for.
2. **EDA** — churn distribution, contract type, price sensitivity, Fiber optic segment,
   and the Tech Support retention effect.
3. **Feature engineering** — `ServiceCount` (number of add-on services subscribed) and
   `CostPerService` (monthly charge normalized by services used).
4. **Modeling** — Logistic Regression (scaled via a pipeline) and Random Forest, on an
   80/20 stratified split. `customerID` and the target are excluded from the features to
   avoid leakage.
5. **Evaluation** — accuracy, precision/recall/F1 per class, confusion matrices, and
   Random Forest feature importance for driver analysis.

## Next Steps

- Address class imbalance (`class_weight="balanced"` or SMOTE) to improve recall on churners
- Hyperparameter tuning (grid/random search) on both models
- Try a gradient-boosted model (XGBoost / LightGBM) for a stronger non-linear baseline
- Turn the model into a scored customer list / simple retention dashboard

## Run It Locally

```bash
git clone https://github.com/maxencegohin-dev/customer-churn-analysis.git
cd customer-churn-analysis
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
originally published by IBM Sample Data Sets. 7,043 rows, 21 columns, one row per customer.
