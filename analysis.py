"""
House Price Prediction — California Housing
Real data: California Housing dataset, 20,640 houses (1990 US Census).
Source: scikit-learn fetch_california_housing (originally Pace & Barry, 1997).
Target: median house value in a block (in $100,000s).
I built this to learn regression on real data — no fake numbers.
Run: python analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# --- Load real data ---
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.to_csv(os.path.join(DATA, "california_housing.csv"), index=False)
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
print(df.describe().round(2).to_string())
print("Missing values:\n", df.isna().sum().to_string())

# --- Feature engineering (simple, honest) ---
df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
df["population_per_household"] = df["Population"] / df["AveOccup"]
df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
            "AveOccup", "Latitude", "Longitude",
            "rooms_per_household", "bedrooms_ratio", "population_per_household"]
X = df[features]
y = df["MedHouseVal"]  # in $100,000s

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
rows = []
preds = {}
for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    preds[name] = y_pred
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    rows.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4),
                 "RMSE": round(rmse, 4), "CV_R2_mean": round(cv.mean(), 4),
                 "CV_R2_std": round(cv.std(), 4)})
    print(f"{name}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} CV R2={cv.mean():.4f}±{cv.std():.4f}")

results = pd.DataFrame(rows)
results.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

# Feature importance from Random Forest
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop features (Random Forest):\n", importance.round(3).to_string())
importance.to_csv(os.path.join(OUT, "feature_importance.csv"))

with open(os.path.join(OUT, "results.txt"), "w") as f:
    f.write("House Price Prediction — California Housing (20,640 houses)\n")
    f.write("Target: MedHouseVal (median house value, in $100,000s)\n")
    f.write(f"Train: {len(X_train)}, Test: {len(X_test)}, split 80/20, random_state=42\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nTop features (Random Forest):\n")
    f.write(importance.round(3).to_string())
    f.write("\n\nNotes: prices are capped at $500,000 in the dataset. "
            "This is California, USA data — not Bangladesh house prices.\n")

# --- Charts ---
plt.rcParams["figure.figsize"] = (9, 5)
plt.style.use("ggplot")

# 1. Price distribution
plt.figure()
plt.hist(df["MedHouseVal"] * 100000, bins=50, color="#2563eb", edgecolor="white")
plt.title("House Price Distribution (California)")
plt.xlabel("Median House Value ($)")
plt.ylabel("Number of blocks")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "01_price_distribution.png"), dpi=150); plt.close()

# 2. Median income vs price
plt.figure()
plt.scatter(df["MedInc"], df["MedHouseVal"] * 100000, alpha=0.15, s=8, color="#059669")
plt.title("Median Income vs House Price")
plt.xlabel("Median Income ($10,000s)"); plt.ylabel("Median House Value ($)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "02_income_vs_price.png"), dpi=150); plt.close()

# 3. House age vs price
plt.figure()
plt.scatter(df["HouseAge"], df["MedHouseVal"] * 100000, alpha=0.15, s=8, color="#d97706")
plt.title("House Age vs House Price")
plt.xlabel("Median House Age (years)"); plt.ylabel("Median House Value ($)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "03_age_vs_price.png"), dpi=150); plt.close()

# 4. Location scatter (colour = price)
plt.figure()
sc = plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                 cmap="viridis", alpha=0.4, s=6)
plt.colorbar(sc, label="Median House Value ($100k)")
plt.title("Where Are the Expensive Houses? (California map)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "04_location_price.png"), dpi=150); plt.close()

# 5. Correlation heatmap
plt.figure()
corr = df[features + ["MedHouseVal"]].corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "05_correlation_heatmap.png"), dpi=150); plt.close()

# 6. Predicted vs actual (best model = Random Forest)
best = "Random Forest"
plt.figure()
plt.scatter(y_test * 100000, preds[best] * 100000, alpha=0.2, s=8, color="#7c3aed")
lims = [0, 550000]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.title(f"Predicted vs Actual Price — {best}")
plt.xlabel("Actual Price ($)"); plt.ylabel("Predicted Price ($)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_predicted_vs_actual.png"), dpi=150); plt.close()

# 7. Feature importance
plt.figure()
importance.plot(kind="barh", color="#2563eb")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "07_feature_importance.png"), dpi=150); plt.close()

print("Done. Charts saved in outputs/")
