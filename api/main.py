import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# MLflow setup

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("walmart_weekly_sales_forecasting")

#Load + clean 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

for df in [train, test, features, stores]:
    df.columns = df.columns.str.strip()

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
features['Date'] = pd.to_datetime(features['Date'])


def clean_markdown(df):
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace('NA', '0').replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


train = clean_markdown(train)
test = clean_markdown(test)
features = clean_markdown(features)

features_ext = pd.merge(features, stores, on='Store', how='left')
train_df = pd.merge(train, features_ext, on=['Store', 'Date', 'IsHoliday'], how='left')
test_df = pd.merge(test, features_ext, on=['Store', 'Date', 'IsHoliday'], how='left')


#Feature engineering (unchanged)

def engineer_features(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsHoliday'] = df['IsHoliday'].map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
    df = pd.get_dummies(df, columns=['Type'], prefix='StoreType')
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

train_sorted = train_engineered.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
grouped = train_sorted.groupby(['Store', 'Dept'])

for lag in [1, 2, 3]:
    train_sorted[f'Sales_Lag_{lag}'] = grouped['Weekly_Sales'].shift(lag)

train_sorted['Sales_MA_4'] = grouped['Weekly_Sales'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
train_sorted['Sales_MA_8'] = grouped['Weekly_Sales'].transform(
    lambda x: x.shift(1).rolling(8, min_periods=1).mean()
)

train_sorted = train_sorted.dropna(subset=['Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3'])

last_sales = train_sorted.groupby(['Store', 'Dept']).last()[['Weekly_Sales']].reset_index()
last_sales = last_sales.rename(columns={'Weekly_Sales': 'Last_Sales'})

test_sorted = test_engineered.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
test_sorted = pd.merge(test_sorted, last_sales, on=['Store', 'Dept'], how='left')
for col in ['Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_MA_4', 'Sales_MA_8']:
    test_sorted[col] = test_sorted['Last_Sales']
test_sorted = test_sorted.fillna(0)

exclude = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Last_Sales']
features_cols = [col for col in train_sorted.columns if col not in exclude]
for col in features_cols:
    if col not in test_sorted.columns:
        test_sorted[col] = 0

#split on an actual date cutoff
#not a positional slice

unique_dates = np.sort(train_sorted['Date'].unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = train_sorted['Date'] < split_date
X_train_split = train_sorted.loc[train_mask, features_cols]
y_train_split = train_sorted.loc[train_mask, 'Weekly_Sales']
X_val = train_sorted.loc[~train_mask, features_cols]
y_val = train_sorted.loc[~train_mask, 'Weekly_Sales']

print(f"Split date: {split_date}")
print(f"Training set: {X_train_split.shape[0]} rows")
print(f"Validation set: {X_val.shape[0]} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)

X_test = test_sorted[features_cols]
string_cols = X_test.select_dtypes(include=['object']).columns.tolist()
for col in string_cols:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
X_test_scaled = scaler.transform(X_test)

#Models to compare

def build_models(args):
    """Build models using hyperparameters supplied from the command line."""
    return {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Ridge': Ridge(alpha=args.ridge_alpha),
        'Lasso': Lasso(alpha=args.lasso_alpha, max_iter=args.lasso_max_iter),
        'Decision Tree': DecisionTreeRegressor(
            random_state=42, max_depth=args.dt_max_depth,
            min_samples_split=args.dt_min_samples_split
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            random_state=42, n_estimators=args.gb_n_estimators,
            max_depth=args.gb_max_depth, learning_rate=args.gb_learning_rate,
            subsample=args.gb_subsample
        ),
        'XGBoost': XGBRegressor(
            random_state=42, n_jobs=-1, n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
            tree_method='hist', subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree
        ),
        'LightGBM': LGBMRegressor(
            random_state=42, n_jobs=-1, n_estimators=args.lgbm_n_estimators,
            num_leaves=args.lgbm_num_leaves, max_depth=args.lgbm_max_depth,
            learning_rate=args.lgbm_learning_rate, subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            verbose=-1, force_row_wise=True
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walmart forecasting with CLI hyperparameters and MLflow"
    )
    parser.add_argument("--model", default="all",
                        choices=["all", "linear", "ridge", "lasso", "decision_tree",
                                 "random_forest", "gradient_boosting", "xgboost", "lightgbm"])
    parser.add_argument("--cv", type=int, default=2)

    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lasso-alpha", type=float, default=1.0)
    parser.add_argument("--lasso-max-iter", type=int, default=1000)

    parser.add_argument("--dt-max-depth", type=int, default=20)
    parser.add_argument("--dt-min-samples-split", type=int, default=100)

    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    parser.add_argument("--rf-min-samples-split", type=int, default=10)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")

    parser.add_argument("--gb-n-estimators", type=int, default=100)
    parser.add_argument("--gb-max-depth", type=int, default=5)
    parser.add_argument("--gb-learning-rate", type=float, default=0.1)
    parser.add_argument("--gb-subsample", type=float, default=0.8)

    parser.add_argument("--xgb-n-estimators", type=int, default=30)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=1.0)

    parser.add_argument("--lgbm-n-estimators", type=int, default=100)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.1)
    parser.add_argument("--lgbm-subsample", type=float, default=1.0)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=1.0)
    return parser.parse_args()


args = parse_args()
models = build_models(args)

model_name_map = {
    "linear": "Linear Regression", "ridge": "Ridge", "lasso": "Lasso",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost",
    "lightgbm": "LightGBM"
}
if args.model != "all":
    selected_name = model_name_map[args.model]
    models = {selected_name: models[selected_name]}



def log_model_by_flavor(name, model):
    """Log with the right MLflow flavor so it round-trips cleanly."""
    if name == 'XGBoost':
        mlflow.xgboost.log_model(model, artifact_path="model")
    elif name == 'LightGBM':
        mlflow.lightgbm.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")


# 5. Comparison loop — one MLflow run per model

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", name)
        mlflow.log_params({f"param_{k}": v for k, v in model.get_params().items()})
        mlflow.log_param("n_train_rows", X_train_split.shape[0])
        mlflow.log_param("n_val_rows", X_val.shape[0])
        mlflow.log_param("n_features", X_train_split.shape[1])
        mlflow.log_param("split_date", str(split_date))

        start_time = time.time()
        cv_scores = cross_val_score(model, X_train_scaled, y_train_split, cv=args.cv, scoring='r2')
        cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

        model.fit(X_train_scaled, y_train_split)
        y_pred = model.predict(X_val_scaled)

        val_r2 = r2_score(y_val, y_pred)
        val_mse = mean_squared_error(y_val, y_pred)
        val_mae = mean_absolute_error(y_val, y_pred)
        fit_time = time.time() - start_time

        mlflow.log_metrics({
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "val_r2": val_r2,
            "val_mse": val_mse,
            "val_rmse": np.sqrt(val_mse),
            "val_mae": val_mae,
            "fit_time_seconds": fit_time,
        })

        log_model_by_flavor(name, model)

        results.append({
            'Model': name, 'CV_R2': cv_mean, 'CV_Std': cv_std,
            'Val_R2': val_r2, 'Val_MSE': val_mse, 'Val_MAE': val_mae,
            'Fit_Time': fit_time, 'Model_Object': model,
        })

        print(f"{name:<20} CV: {cv_mean:.4f}±{cv_std:.4f} | Val R²: {val_r2:.4f} | {fit_time:.2f}s")

results_df = pd.DataFrame(results).sort_values('Val_R2', ascending=False)
print("\n", results_df[['Model', 'CV_R2', 'Val_R2', 'Val_MSE', 'Val_MAE', 'Fit_Time']].to_string(index=False))

best_row = results_df.iloc[0]
best_name = best_row['Model']
best_model = best_row['Model_Object']
print(f"\nBest model: {best_name} (Val R² = {best_row['Val_R2']:.4f})")

#run


with mlflow.start_run(run_name=f"{best_name}_champion"):
    mlflow.set_tag("model_family", best_name)
    mlflow.set_tag("stage", "champion")

    best_model.fit(X_train_scaled, y_train_split)
    y_pred_val = best_model.predict(X_val_scaled)

    mlflow.log_metrics({
        "val_r2": r2_score(y_val, y_pred_val),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
        "val_mae": mean_absolute_error(y_val, y_pred_val),
    })

    test_predictions = np.maximum(best_model.predict(X_test_scaled), 0)
    submission = pd.DataFrame({
        'Id': test_sorted['Store'].astype(str) + '_' +
              test_sorted['Dept'].astype(str) + '_' +
              test_sorted['Date'].dt.strftime('%Y-%m-%d'),
        'Weekly_Sales': test_predictions,
    })
    submission.to_csv('submission.csv', index=False)
    mlflow.log_artifact('submission.csv')

    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({
            'Feature': features_cols,
            'Importance': best_model.feature_importances_,
        }).sort_values('Importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='skyblue')
        ax.set_title(f'{best_name}: Top 15 Feature Importances')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        fig.savefig('feature_importance.png', dpi=150)
        mlflow.log_artifact('feature_importance.png')
        plt.close(fig)

    log_model_by_flavor(best_name, best_model)

