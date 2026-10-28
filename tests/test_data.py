from ai4i_mlops.config import FEATURES, TARGET
from ai4i_mlops.data import load_data, split_data


def test_load_data():
    df = load_data()

    assert len(df) == 10000
    assert all(col in df.columns for col in FEATURES + [TARGET])
    assert not df.isnull().any().any()


def test_split_data():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) == 8000
    assert len(X_test) == 2000

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}