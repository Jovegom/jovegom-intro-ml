import pandas as pd
from sklearn.preprocessing import StandardScaler

def detectar_missing_ocultos(df, valores_sospechosos=[-1, 0, 999, 9999, -999]):
    reporte = {}

    for col in df.columns:
        info = {}

        # Missing reales
        info["NaN reales"] = df[col].isna().sum()

        # Detección de valores no numéricos en columnas "numéricas"
        col_forzada = pd.to_numeric(df[col], errors="coerce")
        info["Valores no numéricos ocultos"] = col_forzada.isna().sum() - df[col].isna().sum()

        # Cantidad de valores únicos 
        info["Valores únicos"] = df[col].nunique()

        reporte[col] = info

    return pd.DataFrame(reporte).T




def encode_categorical(X_train, X_test):
    """
    Codifica variables categóricas:
    - One-hot para columnas con <=4 categorías
    - Frequency encoding para columnas con >4 categorías
    Devuelve X_train_final, X_test_final
    """

    # Separar columnas categóricas y numéricas
    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(exclude=['object']).columns

    # Dividir según cardinalidad
    num_categories = X_train[cat_cols].nunique()
    onehot_cols = num_categories[num_categories <= 4].index.tolist()
    freq_cols   = num_categories[num_categories > 4].index.tolist()

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    X_train_oh = pd.get_dummies(X_train[onehot_cols], drop_first=True, dtype=int)
    X_test_oh  = pd.get_dummies(X_test[onehot_cols], drop_first=True, dtype=int)
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    # -----------------------------
    # Frequency Encoding
    # -----------------------------
    X_train_freq = X_train[freq_cols].copy()
    X_test_freq  = X_test[freq_cols].copy()
    freq_maps = {}
    for col in freq_cols:
        freq = X_train[col].value_counts(normalize=True)
        freq_maps[col] = freq
        X_train_freq[col] = X_train[col].map(freq)
        X_test_freq[col] = X_test[col].map(freq).fillna(0)

    # -----------------------------
    # Combinar con columnas numéricas
    # -----------------------------
    X_train_final = pd.concat(
        [X_train[num_cols].reset_index(drop=True),
         X_train_oh.reset_index(drop=True),
         X_train_freq.reset_index(drop=True)],
        axis=1
    )

    X_test_final = pd.concat(
        [X_test[num_cols].reset_index(drop=True),
         X_test_oh.reset_index(drop=True),
         X_test_freq.reset_index(drop=True)],
        axis=1
    )

    return X_train_final, X_test_final


def scale_numeric_features(X_train, X_test):
    """
    Escala SOLO columnas numéricas.
    No toca variables categóricas codificadas (one-hot / frequency).
    """
    scaler = StandardScaler()

    # Detectar columnas numéricas
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, scaler