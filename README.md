# Ad Conversion Prediction using XGBoost

## Overview

This project predicts whether a customer will convert after interacting with a digital marketing campaign. The model uses customer demographics, campaign details, engagement metrics, and purchase history to classify conversion outcomes.

**Target Variable**

* 1 → Converted
* 0 → Not Converted

---

## Dataset Features

* Customer Information: Age, Gender, Income
* Campaign Information: CampaignChannel, CampaignType, AdSpend, AdvertisingPlatform
* Engagement Metrics: ClickThroughRate, WebsiteVisits, PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks
* Customer History: PreviousPurchases, LoyaltyPoints

---

## Preprocessing

* One-Hot Encoding for categorical features
* Standard Scaling for numerical features
* Stratified Train-Test Split (80:20)
* Class imbalance handled using sample weights

---

## Model

* **Algorithm:** XGBoost Classifier
* **Hyperparameter Tuning:** RandomizedSearchCV (5-Fold Cross Validation)

Tuned parameters include:

* n_estimators
* max_depth
* learning_rate
* gamma
* min_child_weight
* reg_alpha
* reg_lambda

---

## Evaluation

The model was evaluated using the **F1 Score**, making it suitable for handling imbalanced class distributions.

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* XGBoost

---

## Key Learnings

* Data preprocessing with Scikit-Learn Pipelines
* Handling imbalanced datasets
* Hyperparameter tuning
* Model evaluation using classification metrics
* Building an end-to-end machine learning workflow
