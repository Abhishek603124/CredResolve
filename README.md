# CredResolve Intelligence Challenge: Next Best Action Prediction

This repository contains my solution for the CredResolve Intelligence Challenge, a debt recovery fintech competition focused on predicting the success probability of specific collection actions.

##  Problem Overview
CredResolve uses a mix of AI and human intervention to recover debts. The goal is to predict the `TARGET` (probability of success, 0.0 to 1.0) for a given `suggested_action` (Bot Call, Human Call, Field Visit, or Digital Message) based on a borrower's interaction history.

##  Feature Engineering (Customer 360 View)
The raw dataset consisted of fragmented interaction logs. I engineered features to create a comprehensive view of each of the 100,000 borrowers:
* **Teleco (AI Bot):** Extracted sentiment and intent (PTP) from JSON transcripts; calculated call duration and answer ratios.
* **WhatsApp/SMS:** Tracked message delivery, read status, and borrower response sentiment.
* **Human Calls:** Aggregated call counts and agent engagement metrics.
* **Field Visits:** Calculated the success rate of physical visits and analyzed visit outcomes (e.g., "Met Customer").
* **MetaData:** Integrated loan data such as `total_due` and `dpd_bucket`.

##  Modeling Strategy
I utilized a **Stacking Ensemble Regressor** to maximize the R² score, combining the strengths of multiple gradient boosting and tree-based algorithms:

1. **Base Models:** XGBoost, LightGBM, and Random Forest.
2. **Meta-Model:** Ridge Regression used to blend the predictions.
3. **Validation:** 80/20 Train-Validation split with 5-fold cross-validation.

### Performance
* **Initial XGBoost R²:** 0.23
* **Current Ensemble R²:** ~0.30+ (In Progress)

##  How to Run
1. Ensure all CSV files from the challenge are in the `/data` folder.
2. Install dependencies: `pip install pandas scikit-learn xgboost lightgbm catboost`.
3. Run the feature engineering notebook followed by the model training script.
