# StrategiX 2.0 — Analytics Case Competition

**Xavier School of Management (XLRI)**
**Team : 404 Not Found**

> *"How do you save a subscriber who doesn't know they're leaving?"*

This repository contains the complete solution submitted by **Team 404 Not Found** for the StrategiX 2.0 Analytics Case Competition at XLRI. The challenge was to build a predictive model that identifies engagement fatigue in StreamMax OTT platform users — catching at-risk subscribers before they cancel, not after.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Our Approach](#our-approach)
- [Repository Structure](#repository-structure)
- [Dataset Overview](#dataset-overview)
- [Feature Engineering](#feature-engineering)
- [Modelling Pipeline](#modelling-pipeline)
- [Results](#results)
- [Key Insights](#key-insights)
- [Business Strategy](#business-strategy)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Deliverables](#deliverables)

---

## Problem Statement

StreamMax, a growing OTT platform, faces a silent churn problem. By the time traditional models flag a disengaging user, they have already mentally checked out — and re-engaging a churned subscriber costs **5x more** than preventing fatigue in the first place.

The task was to build a binary classifier that predicts **engagement fatigue** (label `1` = fatigued, `0` = engaged) using behavioural usage data, evaluated on **AUC-ROC**.

| Metric | Value |
|---|---|
| Training users | 8,000 |
| Test users | 2,000 |
| Total users | 10,000 |
| Fatigued (label = 1) | 35.4% |
| Engaged (label = 0) | 64.6% |
| Evaluation metric | AUC-ROC |

---

## Our Approach

We reframed the problem from **reactive churn detection** to **proactive fatigue detection** — identifying users 30 days before they cancel rather than after.

```
Raw Usage Data (12 features)
        |
        v
Feature Engineering  -->  33 behavioral signals
        |
        v
Outlier Capping  -->  Winsorisation at 1st / 99th percentile
        |
        v
+----------+  +-----------+  +--------+  +----------+
| XGBoost  |  | LightGBM  |  |  Rand  |  | CatBoost |
| Level-   |  | Leaf-wise |  | Forest |  | Ordered  |
| wise     |  | boosting  |  | Bagging|  | boosting |
+----------+  +-----------+  +--------+  +----------+
        |           |              |           |
        +-----+-----+--------------+-----------+
              |
              v
   Logistic Regression Meta-Learner
   (learns optimal weights per model)
              |
              v
   Fatigue Probability per user  [0.0 --> 1.0]
```

---

## Repository Structure

```
StrategiX-2.0---Analytics-Case-Competition/
|
|-- Data/
|   |-- ott_train.csv               # 8,000 training users with labels
|   |-- ott_test.csv                # 2,000 test users without labels
|
|-- results/
|   |-- oof_predictions.csv         # Out-of-fold predictions (cross-validated)
|   |-- model_comparison.csv        # AUC scores across all model versions
|
|-- catboost_info/                  # CatBoost training logs and metadata
|
|-- visualisations/                 # All EDA and model plots
|   |-- class_distribution.png
|   |-- feature_distributions.png
|   |-- boxplots.png
|   |-- cdf_days_since_last_session.png
|   |-- fatigue_rate_by_tier.png
|   |-- fatigue_rate_by_tenure.png
|   |-- correlation_heatmap.png
|   |-- feature_importance.png
|   |-- shap_importance.png
|
|-- 404_Not_Found_Analysis.ipynb    # Complete modelling notebook
|-- 404_Not_Found_Predictions.csv   # Final submission file
|-- 404_Not_Found_Presentation.pdf  # 6-slide business presentation
|-- TeamName_Predictions.csv        # Alternate submission file
|-- StreamMax OTT Platform.pdf      # Original problem statement
|-- README.md
```

---

## Dataset Overview

### Raw Features (12 columns)

| Feature | Type | Description |
|---|---|---|
| `user_id` | ID | Unique user identifier |
| `tenure_days` | Numeric | Days since account creation |
| `subscription_tier` | Categorical | Basic / Standard / Premium |
| `avg_daily_minutes_last_7d` | Numeric | Average daily watch time — last 7 days |
| `avg_daily_minutes_last_30d` | Numeric | Average daily watch time — last 30 days |
| `sessions_last_7d` | Numeric | Login sessions in last 7 days |
| `sessions_last_30d` | Numeric | Login sessions in last 30 days |
| `avg_completion_rate` | Numeric | Fraction of content watched before exit |
| `days_since_last_session` | Numeric | Days since most recent login |
| `binge_sessions_last_30d` | Numeric | Sessions with 3+ hours of continuous watch |
| `recommendation_click_rate` | Numeric | Fraction of recommended content clicked |
| `unique_genres_watched_30d` | Numeric | Number of distinct genres watched |
| `peak_hour_viewing_pct` | Numeric | Fraction of sessions during peak hours |
| `original_content_pct` | Numeric | Fraction of watch time on original content |
| `fatigue_label` | Binary | Target — 0 = Engaged, 1 = Fatigued |

### Class Distribution

```
Engaged  (0) : 5,169 users  [64.6%]  ████████████████████████
Fatigued (1) : 2,831 users  [35.4%]  █████████████
```

---

## Feature Engineering

We engineered **33 features** from the original 12. Key categories:

### Trend Features
Capture whether engagement is improving or declining relative to the user's own baseline.

| Feature | Formula | Interpretation |
|---|---|---|
| `minutes_trend_ratio` | `avg_7d / avg_30d` | < 1.0 means watching less recently |
| `sessions_trend_ratio` | `sessions_7d / sessions_30d` | < 1.0 means logging in less |
| `minutes_trend` | `avg_7d - avg_30d` | Absolute change in watch time |
| `sessions_trend` | `sessions_7d - sessions_30d` | Absolute change in sessions |

### Engagement Quality Features
Capture satisfaction, not just quantity.

| Feature | Formula | Interpretation |
|---|---|---|
| `engagement_quality` | `completion_rate * click_rate` | Both must be high for true engagement |
| `min_per_session_7d` | `total_7d / sessions_7d` | Session depth — are they staying? |
| `min_per_session_30d` | `total_30d / sessions_30d` | Long-term session depth |
| `binge_ratio` | `binge / sessions_30d` | Fraction of sessions that are deep dives |

### Recency × Activity Interactions
These interaction features were the strongest predictors in the model.

| Feature | Formula | Why it matters |
|---|---|---|
| `recency_x_sessions` | `days_since * sessions_7d` | High days-since + low sessions = danger |
| `recency_x_minutes` | `days_since * avg_7d` | High recency + low watch time = alarm |
| `completion_x_clicks` | `completion * click_rate` | Both signals failing simultaneously |

### Composite Risk Score
```
fatigue_score = (
    0.4 * days_since_last_session_normalised
  + 0.3 * (1 - avg_completion_rate)
  + 0.3 * (1 - recommendation_click_rate)
)
```
Single most predictive feature — Pearson correlation with fatigue label: **+0.43**

---

## Modelling Pipeline

### Why Tree-Based Models

Tree-based gradient boosting was selected over neural networks for this dataset because:

- Dataset size (8,000 rows) is too small for deep learning to outperform trees
- No spatial or sequential structure in features — trees are optimal for tabular data
- Native feature importance enables direct business interpretation
- Research consensus (Grinsztajn et al. 2022) confirms trees outperform neural networks on structured tabular data

### Base Models

| Model | Algorithm | Key Differentiator |
|---|---|---|
| **XGBoost** | Gradient boosting, level-wise tree growth | Strong regularisation, handles sparse features |
| **LightGBM** | Gradient boosting, leaf-wise tree growth | Faster, better on large feature sets |
| **Random Forest** | Bagging, independent trees | Genuinely different error structure from boosting |
| **CatBoost** | Ordered boosting, symmetric trees | Prevents target leakage during gradient computation |

### Validation Strategy

**Stratified 5-Fold Cross-Validation with Out-of-Fold predictions**

```
Fold 1: [Train: 2-5] [Validate: 1]  -->  OOF predictions for fold 1 users
Fold 2: [Train: 1,3-5] [Validate: 2]  -->  OOF predictions for fold 2 users
Fold 3: [Train: 1-2,4-5] [Validate: 3]  -->  OOF predictions for fold 3 users
Fold 4: [Train: 1-3,5] [Validate: 4]  -->  OOF predictions for fold 4 users
Fold 5: [Train: 1-4] [Validate: 5]  -->  OOF predictions for fold 5 users
                                              |
                                              v
                                 Full leakage-free AUC estimate
                                 across all 8,000 training users
```

Each user's risk score was generated *only* by a model that had never seen that user during training. This makes the OOF AUC a clean, leakage-free estimate of generalisation performance.

### Stacking Architecture

```
Level 1 Base Models (4 models, each with 5-fold OOF predictions)
  |
  |-- XGBoost OOF predictions      [8000 x 1]
  |-- LightGBM OOF predictions     [8000 x 1]
  |-- RandomForest OOF predictions [8000 x 1]
  |-- CatBoost OOF predictions     [8000 x 1]
  |
  v
Meta-Feature Matrix                [8000 x 4]
  |
  v
Logistic Regression Meta-Learner
  |-- Learns: how much to trust each model for each user type
  |-- Coefficients: XGB +0.13 | LGB +0.31 | RF +0.18 | CAT +0.61
  |
  v
Final Fatigue Probability          [0.0 --> 1.0]
```

CatBoost received the highest trust coefficient (+0.61), confirming it added the most unique signal not captured by the other three models.

### Hyperparameter Optimisation

Hyperparameters were tuned using **Optuna** (Bayesian optimisation with TPE sampler), 50 trials per model:

| Parameter | Search Range | Why |
|---|---|---|
| `n_estimators` | 300 – 1500 | More trees with lower learning rate generalises better |
| `learning_rate` | 0.005 – 0.1 (log scale) | Smaller steps = more stable convergence |
| `max_depth` | 3 – 8 | Deeper trees can overfit on 8,000 rows |
| `subsample` | 0.6 – 1.0 | Row subsampling reduces overfitting |
| `colsample_bytree` | 0.6 – 1.0 | Feature subsampling increases diversity |
| `reg_alpha` | 0.0 – 1.0 | L1 regularisation — zeroes out noisy features |
| `reg_lambda` | 0.0 – 1.0 | L2 regularisation — shrinks all weights |

---

## Results

### Model Progression

| Stage | Model | OOF AUC | Delta |
|---|---|---|---|
| Baseline | XGBoost (default) | 0.77926 | — |
| Baseline | LightGBM (default) | 0.78715 | +0.00789 |
| Manual tuning | XGBoost (lr=0.02, 1000 trees) | 0.78210 | +0.00284 |
| Manual tuning | LightGBM (lr=0.02, 1000 trees) | 0.78653 | — |
| Ensemble | XGB + LGB + RF (weighted avg) | 0.78830 | +0.00177 |
| + Outlier capping | Winsorised features | 0.78854 | +0.00024 |
| + Optuna tuning | All 4 models tuned | 0.78964 | +0.00110 |
| **Stacking** | **4-model + LR meta-learner** | **0.79031** | **+0.00067** |

### Final Submission Statistics

| Metric | Value |
|---|---|
| **OOF AUC (CV estimate)** | **0.790 ± 0.011** |
| Validation method | Stratified 5-Fold Out-of-Fold |
| Test users predicted | 2,000 |
| Min predicted probability | 0.081 |
| Max predicted probability | 0.870 |
| Mean predicted probability | 0.360 |
| Actual fatigue rate (train) | 0.354 |

> **Note:** The OOF AUC of 0.790 is a cross-validated estimate computed on training data. The true test AUC will be determined by competition evaluation against held-out test labels. The mean prediction (0.360) closely matching the actual fatigue rate (0.354) confirms the stacked model is well-calibrated.

---

## Key Insights

### Insight 1 — Recency Is the #1 Warning Signal

| Group | Logged in Today | 7-day Activity |
|---|---|---|
| Engaged users | **40%** | High |
| Fatigued users | **28%** | Declining |

Once a user crosses **7 days of inactivity**, the probability of fatigue classification jumps sharply. The daily login habit is the clearest observable signal of disengagement.

### Insight 2 — They Watch Less *and* Enjoy It Less

| Metric | Engaged | Fatigued | Drop |
|---|---|---|---|
| Avg completion rate | 0.425 | 0.315 | **-26%** |
| Recommendation click rate | 0.231 | 0.157 | **-32%** |

Fatigued users are not simply watching less — they are opening content and abandoning it. The drop in recommendation click rate (-32%) signals that they have stopped trusting the platform to surface content worth watching.

### Insight 3 — Basic Tier and New Users Are Highest Risk

| Segment | Fatigue Rate |
|---|---|
| Basic tier | **38.2%** |
| Standard tier | 35.5% |
| Premium tier | 31.1% |
| Users aged 0-30 days | **39.2%** |
| Users aged 365d+ | 31.7% |

New users have the highest fatigue rate of any group — they churn before finding their content identity on the platform.

### Top Predictors — Model Confirmed

| Rank | Feature | Business Meaning | Correlation |
|---|---|---|---|
| #1 | `fatigue_score` | All risk signals combined | +0.43 |
| #2 | `avg_daily_minutes_last_7d` | Watching less than usual | -0.30 |
| #3 | `avg_completion_rate` | Starting but not finishing content | -0.29 |
| #4 | `recommendation_click_rate` | Stopped trusting the algorithm | -0.28 |
| #5 | `days_since_last_session` | Daily habit breaking down | +0.26 |

---

## Business Strategy

### User Segments at Risk

| # | Segment | Profile | Fatigue Rate | Share of Risk |
|---|---|---|---|---|
| 01 | Silent Drifter | Standard/Basic, 90-180d tenure, sessions declining | Medium-High | ~28% |
| 02 | Lapsed Binge Watcher | Any tier, 3+ binges/month → 0 this month | Very High | ~18% |
| 03 | New User Dropout | 0-30 days tenure, never onboarded | **Critical** | ~15% |
| 04 | Dissatisfied Viewer | Completion <0.25, click rate <0.10 | High | ~20% |
| 05 | Price-Sensitive Basic | Basic tier, comparing with competitors | Medium | ~19% |

### Six Strategic Recommendations

**REC 01 — Deploy Fatigue Score Dashboard**
Compute the model's fatigue score for every user daily. Segment users into Red (score >0.65), Amber (0.40-0.65), and Green (<0.40). Red users trigger automatic outreach within 24 hours.

**REC 02 — Personalised Re-engagement for Silent Drifters**
When a user's 7-day watch time drops more than 40% vs their 30-day baseline, trigger a push notification with content recommendations based on their top genre — not a generic top-10 list.

**REC 03 — Fix the Onboarding Window (Day 0-30)**
Implement a mandatory taste profile setup in the first session. Use 5 genre preferences and 3 format preferences to seed the first 2 weeks of recommendations exclusively, giving new users immediate content-fit.

**REC 04 — Binge Recovery Campaign**
When a user who had 3+ binge sessions last month shows zero binge sessions this month, trigger a "Continue your journey" notification referencing their last series and surfacing similar content.

**REC 05 — Tier Upgrade Incentive for At-Risk Basic Users**
Offer Amber-risk Basic tier users a 2-month Standard upgrade at 50% off. The 7-point fatigue rate gap between Basic (38.2%) and Premium (31.1%) is directly linked to content access limitations.

**REC 06 — Recommendation Algorithm Reset**
When a user's click rate falls below 0.10 for 2+ consecutive weeks, temporarily switch from collaborative filtering to genre-popularity-based recommendations within their historically preferred genres, breaking the feedback loop.

### 90-Day Implementation Roadmap

```
Week 1-2           Month 1-2          Month 2-3          Ongoing
DEPLOY             TEST               SCALE              MONITOR
  |                  |                  |                  |
Deploy fatigue     A/B test nudges    Tier upgrade       Retrain model
score pipeline     vs control         campaign           monthly
                                                        
Build Red/Amber/   Revised            Binge Recovery     Track KPIs:
Green dashboard    onboarding for     campaign           - Fatigue rate
                   new users                             - Conversion
Flag all Red       Measure causal     Fatigue score      - Revenue/user
users for          lift from          integrated
outreach           interventions      into rec algo
```

---

## How to Run

### 1 — Clone the repository

```bash
git clone https://github.com/nikhilchaudhary7108/StrategiX-2.0---Analytics-Case-Competition.git
cd StrategiX-2.0---Analytics-Case-Competition
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap matplotlib seaborn
```

### 3 — Run the notebook

Open `404_Not_Found_Analysis.ipynb` in Jupyter and run all cells sequentially. The notebook is structured in the following order:

```
Section 1  -->  Data loading and EDA
Section 2  -->  Feature engineering (33 features)
Section 3  -->  Outlier capping (Winsorisation)
Section 4  -->  Baseline models (XGBoost, LightGBM, Random Forest)
Section 5  -->  Hyperparameter tuning with Optuna
Section 6  -->  CatBoost training
Section 7  -->  Stacking ensemble (Logistic Regression meta-learner)
Section 8  -->  Final predictions and submission export
```

### 4 — Output

The final predictions file `404_Not_Found_Predictions.csv` will be generated with the following format:

| user_id | predicted_fatigue_probability |
|---|---|
| 8001 | 0.3412 |
| 8002 | 0.7823 |
| ... | ... |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | >= 1.5 | Data manipulation |
| `numpy` | >= 1.23 | Numerical operations |
| `scikit-learn` | >= 1.2 | Random Forest, Logistic Regression, CV utilities |
| `xgboost` | >= 1.7 | XGBoost classifier |
| `lightgbm` | >= 3.3 | LightGBM classifier |
| `catboost` | >= 1.2 | CatBoost classifier |
| `optuna` | >= 3.0 | Bayesian hyperparameter optimisation |
| `shap` | >= 0.41 | Feature importance and interaction values |
| `matplotlib` | >= 3.6 | Visualisations |
| `seaborn` | >= 0.12 | Statistical plots |

---

## Deliverables

| File | Description |
|---|---|
| [`404_Not_Found_🚫_Analysis.ipynb`](./404_Not_Found_🚫_Analysis.ipynb) | Complete modelling notebook with EDA, feature engineering, training, tuning, and stacking |
| [`404_Not_Found_🚫_Predictions.csv`](./404_Not_Found_🚫_Predictions.csv) | Final submission — fatigue probability for all 2,000 test users |
| [`404_Not_Found_🚫_Presentation.pdf`](./404_Not_Found_🚫_Presentation.pdf) | 6-slide business presentation covering insights, model, and strategy |
| [`visualisations/`](./visualisations/) | All EDA plots, feature importance charts, and SHAP analysis |

---

## Model Limitations

| Limitation | What It Means | How We Address It |
|---|---|---|
| OOF AUC 0.790 is a CV estimate | True test AUC pending evaluation — may differ | Reported as estimate, not verified score |
| 21% of predictions are incorrect | Model misclassifies roughly 1 in 5 users | Human review recommended for high-value Premium users |
| Correlation is not causation | High fatigue score does not guarantee an intervention will work | A/B test every campaign before scaling |
| Model requires periodic retraining | User behaviour shifts with content library and seasons | Monthly retraining cycle planned in Phase 4 |

---

## Authors

**Team 404 Not Found**
StrategiX 2.0 — Analytics Case Competition
Xavier School of Management (XLRI)

---

*All predictions, models, and analysis in this repository were developed solely for the StrategiX 2.0 competition using the provided StreamMax dataset.*
