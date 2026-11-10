"""
CodeAlpha Data Science Internship - Task 3: Car Price Prediction with Machine Learning
--------------------------------------------------------------------------------------
Predicts car prices from features such as brand, engine size, horsepower,
mileage and age. Compares several regression models, tunes the best
tree-based one, and explains which features drive price.

Usage:
    python car_price_prediction.py --csv CarPrice_Assignment.csv
    python car_price_prediction.py --csv "car data.csv"

Works with both common versions of the dataset (the target column and the
car-name / year columns are detected automatically).
"""

import argparse
import datetime
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TARGET_CANDIDATES = ["price", "selling_price", "sellingprice"]
BRAND_FIXES = {"maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota",
               "vokswagen": "volkswagen", "vw": "volkswagen",
               "alfa-romero": "alfa-romeo"}

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS
# ----------------------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect the target column
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"No price column found. Columns: {list(df.columns)}")

    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape} | Target: '{target}'")
    print("\nMissing values:")
    miss = df.isnull().sum()
    print(miss[miss > 0].to_string() if miss.any() else "None")

    # Drop ID-like columns, duplicates, and rows with no target
    df = df.drop(columns=[c for c in df.columns
                          if c in ("id", "car_id") or c.startswith("unnamed")])
    df = df.drop_duplicates().dropna(subset=[target])

    # Feature engineering: brand from the car name
    name_col = next((c for c in ("carname", "car_name") if c in df.columns), None)
    if name_col:
        brand = (df[name_col].astype(str).str.lower().str.strip()
                 .str.split().str[0].replace(BRAND_FIXES))
        if brand.nunique() <= 50:
            df["brand"] = brand
            print(f"\nExtracted 'brand' from '{name_col}' ({brand.nunique()} brands)")
        df = df.drop(columns=name_col)  # raw names are too specific to generalise

    # Feature engineering: car age from manufacturing year
    if "year" in df.columns:
        df["car_age"] = datetime.datetime.now().year - df["year"]
        df = df.drop(columns="year")
        print("Created 'car_age' from 'year'")

    return df, target


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return pre, num_cols, cat_cols


def make_model(preprocessor, regressor):
    """Pipeline + log-transformed target (prices are right-skewed)."""
    pipe = Pipeline([("pre", preprocessor), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


# ----------------------------------------------------------------------------
# 2. EXPLORE
# ----------------------------------------------------------------------------
def explore(df, target):
    print("\nSummary statistics:")
    print(df.describe().round(2).T.head(15))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[target], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Price Distribution")
    sns.histplot(np.log1p(df[target]), kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Log-Transformed Price")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), dpi=150)
    plt.close(fig)

    num = df.select_dtypes(include="number")
    if num.shape[1] > 2:
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=num.shape[1] <= 12, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"\nTop correlations with {target}:")
        print(corr[target].drop(target).sort_values(key=abs, ascending=False)
              .head(6).round(2).to_string())

    if "brand" in df.columns:
        order = df.groupby("brand")[target].median().sort_values(ascending=False).index[:12]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[df["brand"].isin(order)], x="brand", y=target,
                    order=order, ax=ax)
        ax.set_title("Price by Brand (top 12 by median price)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "price_by_brand.png"), dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------
# 3. TRAIN & EVALUATE
# ----------------------------------------------------------------------------
def scores(model, X_test, y_test):
    pred = model.predict(X_test)
    return {"R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))}


def compare_models(pre, X_train, X_test, y_train, y_test):
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300,
                                               random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    fitted, rows = {}, []
    for name, reg in candidates.items():
        model = make_model(pre, reg)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({"Model": name, "CV R2": cv.mean(),
                     **scores(model, X_test, y_test)})

    # Light hyperparameter tuning for Gradient Boosting
    print("\nTuning Gradient Boosting (small grid search)...")
    grid = {"regressor__model__n_estimators": [200, 400],
            "regressor__model__max_depth": [2, 3, 4],
            "regressor__model__learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(make_model(pre, GradientBoostingRegressor(random_state=RANDOM_STATE)),
                      grid, cv=3, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best params:", {k.split("__")[-1]: v for k, v in gs.best_params_.items()})
    fitted["Gradient Boosting (tuned)"] = gs.best_estimator_
    rows.append({"Model": "Gradient Boosting (tuned)", "CV R2": gs.best_score_,
                 **scores(gs.best_estimator_, X_test, y_test)})

    results = (pd.DataFrame(rows).sort_values("R2", ascending=False)
               .reset_index(drop=True))
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(results.round(3).to_string(index=False))
    results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=results, x="R2", y="Model", hue="Model", legend=False,
                palette="viridis", ax=ax)
    ax.set_title("Model Comparison - Test R²")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    return fitted, results


def diagnostics(model, name, X_test, y_test):
    pred = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, pred, alpha=0.7, color="#4C72B0")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[0].plot(lims, lims, "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual price")
    axes[0].set_ylabel("Predicted price")
    axes[0].set_title(f"Actual vs Predicted - {name}")
    axes[0].legend()
    resid = y_test - pred
    axes[1].scatter(pred, resid, alpha=0.7, color="#C44E52")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted price")
    axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residuals")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)


def plot_importance(model, X_test, y_test):
    """Permutation importance works on the raw columns, whatever the model."""
    r = permutation_importance(model, X_test, y_test, n_repeats=10,
                               random_state=RANDOM_STATE, scoring="r2")
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values()
    top = imp.tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", color="#55A868", ax=ax)
    ax.set_title("Top Features Driving Price (permutation importance)")
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("\nMost important features:")
    print(imp.sort_values(ascending=False).head(6).round(3).to_string())


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the car dataset CSV")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, target = load_and_prepare(args.csv)
    explore(df, target)

    X, y = df.drop(columns=target), df[target]
    pre, num_cols, cat_cols = build_preprocessor(X)
    print(f"\nFeatures: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    fitted, results = compare_models(pre, X_train, X_test, y_train, y_test)

    best_name = results.loc[0, "Model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    diagnostics(best, best_name, X_test, y_test)
    plot_importance(best, X_test, y_test)

    joblib.dump(best, os.path.join(OUTPUT_DIR, "best_model.joblib"))

    # Example prediction
    sample = X_test.iloc[[0]]
    print(f"\nExample: actual price {y_test.iloc[0]:,.2f} -> "
          f"predicted {best.predict(sample)[0]:,.2f}")
    print(f"\nAll charts, results and the saved model are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
