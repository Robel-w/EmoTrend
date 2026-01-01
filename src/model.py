import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from features import extract_features

data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed.csv'
df = pd.read_csv(data_path)
X, y = extract_features(df)

# Train multiple models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

best_r2 = -1
best_model = None

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"{name} → R²: {r2:.3f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# Save best model
joblib.dump(best_model, '../models/linear_reg.pkl')
print(f"Best model saved! R² = {best_r2:.3f}")

