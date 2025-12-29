import pandas as pd
from sklearn.model_selection import train_test_split


def csv_to_df(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file at {file_path}: {e}")
        raise

def split_test_train_data():
    """"""
    try:
        # load loan_data.csv into panda DF
        loan_data_df = csv_to_df("data/loan_data.csv")

        seed = 10
        # split input data into train and test
        X_train, X_test, y_train, y_test = train_test_split(loan_data_df.drop("default", axis=1),
                                                            loan_data_df["default"],
                                                            test_size=0.2,
                                                            train_size=0.8,
                                                            random_state=seed)

        print(f"train test shapes = {X_train.shape, X_test.shape}")
        print(f"default rate train, test = {y_train.mean(), y_test.mean()}")
        # print(f"top 1 record from train_data = {X_train.head(1)}")
        # print(f"top 1 record from test_data = {X_test.head(1)}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error: {e}")