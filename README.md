# CodeAlpha_CarPricePrediction

**CodeAlpha Data Science Internship - Task 3**

Predicts used/new car prices from features such as brand, engine size, horsepower, mileage and age using machine learning regression.

## What this project does
1. **Loads & cleans** the data (removes duplicates, handles missing values, drops ID columns)
2. **Feature engineering**
   - `brand` extracted from the car name (with typo fixes, e.g. `maxda` -> `mazda`), capturing brand value
   - `car_age` computed from the manufacturing year (when available)
3. **Explores** price distribution, correlations and price by brand
4. **Preprocesses** with a scikit-learn pipeline: median imputation + scaling for numbers, one-hot encoding for categories
5. **Trains & compares** Linear Regression, Ridge, Random Forest and Gradient Boosting, plus a tuned Gradient Boosting (grid search). Prices are log-transformed because they are right-skewed
6. **Evaluates** with R², MAE and RMSE, plus actual-vs-predicted and residual plots
7. **Explains** the model with permutation feature importance
8. **Saves** the best model to `outputs/best_model.joblib`

## How to run
```bash
pip install -r requirements.txt
python car_price_prediction.py --csv "your_dataset.csv"
```
Charts, the model comparison table and the saved model are written to `outputs/`.

## Results
*(Fill in from your run: best model, test R², MAE, and the top features from `outputs/model_comparison.csv` and `feature_importance.png`.)*

## Using the saved model
```python
import joblib, pandas as pd
model = joblib.load("outputs/best_model.joblib")
model.predict(new_cars_dataframe)  # same columns as the training data
```

## Limitations
- Small dataset, so results vary with the random split
- Prices depend on market and time; the model should be retrained on fresh data to stay accurate

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
