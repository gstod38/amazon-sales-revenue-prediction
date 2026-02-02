# Amazon Sales Revenue Prediction

## Project Overview
This project analyzes Amazon sales data to identify key drivers of revenue and build regression models that estimate expected order revenue using business-available signals (category, region, payment method, reviews, price, discount, and time-derived features).

## Business Problem
Stakeholders want to understand what factors influence revenue and estimate expected revenue per order to support pricing, promotion planning, and category-level strategy.

## Dataset
Order-level dataset (50,000 rows) with pricing, discounts, category, region, payment method, ratings, reviews, and total revenue.

### Target
- `total_revenue`

### Leakage Handling
To keep the modeling setup realistic, the following post-sale/derived variables were excluded:
- `quantity_sold`
- `discounted_price`

## Approach
- Exploratory Data Analysis (EDA) to understand distributions, trends, and potential drivers
- Preprocessing: date parsing, data hygiene, leakage removal
- Feature engineering: month/year extraction, one-hot encoding for categoricals
- Modeling: baseline + non-linear models
- Evaluation: MAE, RMSE, R² + residual diagnostics

## Results
Model performance on the held-out test set:

| Model | MAE | RMSE | R² |
|------|-----:|-----:|---:|
| Linear Regression | 270.14 | 359.05 | 0.5245 |
| Random Forest | 273.30 | 366.93 | 0.5034 |
| Linear Regression (Log Target) | 294.75 | 397.38 | 0.4176 |
| Gradient Boosting Regressor | **265.37** | **358.20** | **0.5268** |

Model evaluation results were persisted to disk to ensure reproducibility and facilitate comparison across experiments.

**Final model:** Histogram-based Gradient Boosting (best overall MAE/RMSE/R²).

## Key Insights
- **Price is the dominant driver** of predicted revenue.
- **Social proof matters:** review volume and rating add meaningful signal beyond price.
- **Discount percent contributes less than expected** in this dataset, suggesting limited incremental impact on revenue at the order level.
- **Seasonality exists but is mild**, captured through month/year features.

## Limitations
- Order-level revenue is inherently noisy; unobserved drivers (customer intent, marketing context, bundles) limit predictive ceiling.
- No customer-level features (repeat purchases, lifetime value), which would likely improve usefulness.
- No time-lag features (rolling trends), so temporal dynamics are simplified.

## Next Steps
- Reframe as **high-value order classification** (low/medium/high) for more actionable predictions.
- Aggregate to **category-month** or **region-month** revenue prediction to reduce noise.
- Add time-based lag features if historical context is available.
