from typing import Tuple, List
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate(model, name, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    mae = mean_absolute_error(y_v, preds)
    rmse = np.sqrt(mean_squared_error(y_v, preds))
    print(f"=== {name} ===")
    print(f"MAE:  {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD\n")
    return mae, rmse

def OHE(df: pd.DataFrame, columns: List[str], encoder=None) -> Tuple[pd.DataFrame, List[str], OneHotEncoder]:
    index = df.index
    
    if encoder is None:
        # Обучаем кодировщик (для Train) с защитой от мультиколлинеарности
        encoder = OneHotEncoder(sparse_output=False, categories='auto', drop='first', handle_unknown='ignore')
        ohe_matrix = encoder.fit_transform(df[columns])
    else:
        # Только трансформируем (для Valid)
        ohe_matrix = encoder.transform(df[columns])
        
    col_names = encoder.get_feature_names_out(columns)
    
    df_dropped = df.drop(columns, axis=1).reset_index(drop=True)
    df_ohe = pd.DataFrame(ohe_matrix, columns=col_names)
    
    res_df = pd.concat([df_dropped, df_ohe], axis=1)
    res_df.index = index
    return res_df, list(col_names), encoder

current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)           
data_path = os.path.join(project_root, 'data', 'diamonds_cleaned.csv')

df_final = pd.read_csv(data_path)
print(f"Очищенный датасет успешно загружен! Размер: {df_final.shape}")

# Заполняем пропуски в fluor строкой 'NONE' 
if 'fluor' in df_final.columns:
    df_final['fluor'] = df_final['fluor'].fillna('NONE')

X = df_final.drop(columns=['id', 'price', 'price_per_carat', 'price_per_carat_adjusted', 'date'])
y = df_final['price_per_carat_adjusted']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=13)

X_train_proc, ohe_features, trained_ohe = OHE(X_train, cat_cols)
X_valid_proc, _, _ = OHE(X_valid, cat_cols, encoder=trained_ohe)

scaler = StandardScaler()

X_train_proc[num_cols] = scaler.fit_transform(X_train_proc[num_cols])
X_valid_proc[num_cols] = scaler.transform(X_valid_proc[num_cols])

summary_metrics = {}

lr_model = LinearRegression()
summary_metrics["Linear Regression (OHE)"] = train_and_evaluate(lr_model, "Линейная регрессия (с OHE и Скейлером)", X_train_proc, X_valid_proc, y_train, y_valid)

knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
summary_metrics["KNN"] = train_and_evaluate(knn_model, "K-Nearest Neighbors (KNN)", X_train_proc, X_valid_proc, y_train, y_valid)

dt_model = DecisionTreeRegressor(max_depth=12, random_state=13)
summary_metrics["Decision Tree"] = train_and_evaluate(dt_model, "Дерево решений (Decision Tree)", X_train_proc, X_valid_proc, y_train, y_valid)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=13, n_jobs=-1)
summary_metrics["Random Forest (Ensemble)"] = train_and_evaluate(rf_model, "Случайный лес (Random Forest)", X_train_proc, X_valid_proc, y_train, y_valid)
