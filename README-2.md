# Payment Transaction Anomaly Classification

## Problem Statement
Classify payment transaction records as routine or anomalous (potentially
fraudulent) using measurable transaction characteristics.

## Dataset
- Credit Card Fraud Detection dataset (European cardholders), originally
  from the ULB Machine Learning Group, mirrored by TensorFlow:
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- 284,807 transactions, 31 columns (Time, Amount, V1–V28 PCA features, Class)
- Highly imbalanced: 492 fraud cases (0.17%) vs 284,315 legitimate

## Approach
1. Loaded and inspected the dataset (no missing values; 1,081 duplicate
   rows found and removed).
2. Explored class balance and transaction amount distribution by class.
3. Scaled `Amount` and `Time` separately using StandardScaler; used the
   PCA-derived V1–V28 features as-is.
4. Split data 80/20 with stratification to preserve the fraud ratio in
   both sets.
5. Trained a Logistic Regression model with `class_weight='balanced'` to
   counter the severe class imbalance.
6. Evaluated using Accuracy, Precision, Recall, F1-score, Confusion
   Matrix, and ROC-AUC — prioritizing Recall and ROC-AUC over raw
   accuracy, since accuracy alone is misleading on imbalanced data.

## Results
| Metric | Value |
|---|---|
| Accuracy | 97.52% |
| Precision | 5.61% |
| Recall | 87.37% |
| F1-score | 10.55% |
| ROC-AUC | 0.966 |

Confusion Matrix:
```
[[55255  1396]
 [   12    83]]
```

## Interpretation
The model correctly identifies 87% of actual fraud cases (high Recall) —
the most important property for a fraud detector, since missing fraud is
far costlier than a false alarm. Precision is low because the model
flags many legitimate transactions as suspicious; in practice, this
trade-off (catch more fraud at the cost of more false alerts) is often
acceptable and can be tuned via the classification threshold. The
ROC-AUC of 0.966 shows strong overall separation between the two classes.

## Streamlit App
`app.py` loads the trained model and scalers and offers two modes:
1. Predict on a real sample transaction from the test set (shows the
   actual label for comparison).
2. Manual entry of Amount and Time, with other features held at neutral
   values, for a quick interactive demo.

## Project Structure
- `notebook.ipynb` — full source code (EDA, cleaning, training, evaluation)
- `fraud_model.pkl`, `amount_scaler.pkl`, `time_scaler.pkl`,
  `feature_columns.pkl` — trained model and preprocessing artifacts
- `sample_transactions.csv` — sample test transactions for the app demo
- `app.py` — Streamlit prediction app
- `requirements.txt` — dependencies
- `README.md` — this file

## How to Run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Limitations
- Precision is low; a production system would need threshold tuning or
  additional techniques (e.g. SMOTE, cost-sensitive learning) to reduce
  false positives.
- V1–V28 are anonymized PCA features, so the manual-entry app mode can't
  meaningfully vary them — it demonstrates the pipeline rather than full
  real-world usability.
- This is an educational prototype, not a production fraud system.

## Future Scope
- Compare against a Decision Tree or Random Forest.
- Apply SMOTE or other resampling techniques to improve Precision.
- Tune the classification threshold based on business cost trade-offs.

## Credits
Dataset: Kaggle / ULB Machine Learning Group, "Credit Card Fraud
Detection Dataset," mirrored by TensorFlow for public use.
