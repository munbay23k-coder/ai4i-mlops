
import pandas as pd
from sklearn.model_selection import train_test_split

from ai4i_mlops.config import (
    DATA_PATH,
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Verificar columnas necesarias
    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Columnas faltantes: {missing_columns}")

    # Seleccionar las variables del modelo
    df = df[required_columns].copy()

    # Verificar valores faltantes
    if df.isnull().any().any():
        raise ValueError("El dataset contiene valores nulos")

    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def main() -> None:
    df = load_data()

    print(f"Datos cargados: {df.shape}")
    print("\nDistribución de clases:")
    print(df[TARGET].value_counts())

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"\nEntrenamiento: {len(X_train)} registros")
    print(f"Prueba: {len(X_test)} registros")


if __name__ == "__main__":
    main()
