# House Price Prediction on the California Housing Dataset

**Author:** Md Emon Rana, Department of CSE, Southeast University, Dhaka
**Data:** California Housing, 20,640 houses (1990 US Census), via scikit-learn `fetch_california_housing`

## I. Introduction

House prices depend on many things at once — income of the area, age of the house, number of rooms, and location. A simple straight-line model often fails because these effects are not linear. This project asks a simple question: how well can we predict a block's median house value from those features, and which feature matters most? I compare three models and report the honest numbers.

## II. Data

The California Housing dataset contains 20,640 blocks, each described by 8 features plus the target median house value (`MedHouseVal`, in $100,000s). Mean house value is 2.07 (≈ $207,000), median 1.80, range 0.15–5.00. There are no missing values in any column. I saved a copy as `data/california_housing.csv`.

I added three engineered features: rooms per household (`AveRooms / AveOccup`), bedrooms ratio (`AveBedrms / AveRooms`), and population per household (`Population / AveOccup`). Infinite values from division were replaced with the column median.

## III. Method

I split the data 80/20 (16,512 train / 4,128 test, `random_state=42`) and scaled features for Linear Regression. Three models were trained: Linear Regression, Random Forest (200 trees), and Gradient Boosting. Each model was also checked with 5-fold cross-validation. I report R², mean absolute error (MAE) and root mean squared error (RMSE) on the held-out test set. All steps are in `analysis.py` and in the Jupyter notebook `analysis.ipynb` (executed, 0 errors).

## IV. Results

| Model | R² | MAE | RMSE | 5-fold CV R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.6535 | 0.4874 | 0.6738 | 0.6593 ± 0.0195 |
| Random Forest | **0.8065** | **0.3280** | **0.5035** | **0.8039 ± 0.0052** |
| Gradient Boosting | 0.7788 | 0.3668 | 0.5384 | 0.7897 ± 0.0026 |

Random Forest is clearly the best: it explains about 81% of the variance in house prices, with an average error of about $32,800 (MAE 0.3280 × $100,000) and RMSE about $50,350. Gradient Boosting is close behind (R² 0.7788); Linear Regression is far worse (R² 0.6535), which shows the relationship is not linear.

Feature importance (Random Forest) is dominated by median income (0.522), followed by average occupancy (0.126), latitude (0.082), longitude (0.082) and house age (0.052). The location scatter shows expensive houses cluster near the coast, around San Francisco and Los Angeles. The predicted-vs-actual plot follows the diagonal well at low and mid prices but flattens at the top.

## V. Limitations

1. **Prices are capped at $500,000** in the dataset, so the model cannot tell very expensive houses apart — this flattens predictions at the top end.
2. **This is California, USA data, not Bangladesh.** The model must not be applied to Dhaka house or rent prices.
3. The target is a block median, not an individual house price.
4. Models were used with near-default settings; tuning could improve the scores a little, but the ranking above is the honest result of this run.

## VI. Conclusion

On real California housing data, Random Forest predicts median house values with R² 0.81 and an average error of about $32,800, clearly beating a linear model. Income and location matter far more than the age or size of the house itself. The full code, notebook, charts and results are in this repository so anyone can re-run it and get the same numbers.

## References

- Pace, R. K. and Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3).
- scikit-learn, California Housing dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
