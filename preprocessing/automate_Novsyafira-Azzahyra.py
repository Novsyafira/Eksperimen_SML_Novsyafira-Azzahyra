import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib # Penting untuk menyimpan scaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(BASE_DIR)
RAW_DATA_PATH = os.path.join(ROOT_DIR, "diabetes_raw", "diabetes.csv") 
OUTPUT_DIR = os.path.join(BASE_DIR, "diabetes_preprocessing")
OUTPUT_DATA_PATH = os.path.join(OUTPUT_DIR, "diabetes_clean.csv")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl") 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")
    return pd.read_csv(file_path)

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)].index.tolist()

def preprocess_data(df):
    print("Mendeteksi outlier...")
    outlier_indices = []
    features = [col for col in df.columns if col != 'Outcome']
    
    for col in features:
        outliers = detect_outliers_iqr(df[col])
        outlier_indices.extend(outliers)

    outlier_indices = list(set(outlier_indices))
    print(f"Jumlah outlier dibuang: {len(outlier_indices)}")
    
    df_clean = df.drop(index=outlier_indices).reset_index(drop=True)
    return df_clean

def scale_and_save(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler berhasil disimpan di: {SCALER_PATH}")
    
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['Outcome'] = y
    
    return df_final

if __name__ == "__main__":
    print("Memulai Otomatisasi Preprocessing...")
    
    # 1. Load
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    df_final = scale_and_save(df_clean)
    
    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Data preprocessing selesai. Disimpan di: {OUTPUT_DATA_PATH}")