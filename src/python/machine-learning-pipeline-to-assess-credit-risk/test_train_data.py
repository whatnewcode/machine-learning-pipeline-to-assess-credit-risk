import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer
from dateutil.relativedelta import relativedelta


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


def impute_missing_numeric_discrete_data(x_train, x_test):
    """
    Impute missing numeric discrete data in training and testing datasets.
    Args:
        x_train (pd.DataFrame): Training feature set.
        x_test (pd.DataFrame): Testing feature set.
    Returns:
        pd.DataFrame, pd.DataFrame: Imputed training and testing feature sets.
    """
    try:
        # numeric columns
        numeric_cols = x_train.select_dtypes(include=['number']).columns
        # discrete columns
        discrete_numeric_cols = [col for col in numeric_cols if x_train[col].nunique() < 20]

        print(f"discrete_numeric_cols ={discrete_numeric_cols}")
        # numeric columns with missing data
        numeric_cols_na = [col for col in numeric_cols if x_train[col].isnull().sum() > 0]

        print(f"numeric columns with missing values {numeric_cols_na}")

        imputer = ArbitraryNumberImputer(
            arbitrary_number=-1,  # the imputation value
            variables=numeric_cols_na,  # the variables to impute
        )

        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)

        # test for missing values for discrete columns
        # print(
            # f"Training data after imputing missing values for numeric columns: \n{x_train[numeric_cols_na].isnull().sum()}")
        # print(
            # f"Test data after imputing missing values for numeric columns: \n{x_test[numeric_cols_na].isnull().sum()}")

        return x_train, x_test

    except Exception as e:
        print(f"Error during imputation: {e}")
        raise


def model_input_features(data):
    """
    data: pandas dataframe
    derives columns/features for model
    :return:
    """
    # total income
    data["total_income"] = (data["income_from_employer"] +
                            data["income_from_pension"] +
                            data["income_from_family_allowance"] +
                            data["income_from_social_welfare"] +
                            data["income_from_leave_pay"] +
                            data["income_from_child_support"] +
                            data["income_other"])
    # print(f"total_income={data['total_income']}")

    # debt to income ratio
    data["debt_income_ratio"] = np.where(data["total_income"] != 0, data["total_debt"] / data["total_income"], 0)
    # data["debt_income_ratio"] = data["total_debt"].div(data["total_income"].mask(data['total_income'] == 0.0)).fillna(0)

    # discretionary income, what's left after paying existing loan
    data["discretionary_income"] = data["total_income"] - data["total_debt"]

    # Age of customer at the time of loan application
    data["age"] = data.apply(lambda row: relativedelta(pd.to_datetime(row["application_date"]), pd.to_datetime(row["date_of_birth"])).years, axis=1)
    # (pd.to_datetime(data["application_date"]).dt.normalize() - pd.to_datetime(data["date_of_birth"]))

    # print(f"dob = {pd.to_datetime(data['date_of_birth']).dt.normalize()}")

    return data


def split_test_train_data():
    """
    splits model data into train and test
    :return:
    """
    try:
        # load loan_data.csv into panda DF
        loan_data_df = csv_to_df("loan_data.csv")

        seed = 10
        # split input data into train and test
        X_train, X_test, y_train, y_test = train_test_split(loan_data_df.drop("default", axis=1),
                                                            loan_data_df["default"],
                                                            test_size=0.2,
                                                            train_size=0.8,
                                                            random_state=seed)

        print(f"train test shapes = {X_train.shape, X_test.shape}")
        print(f"default rate train, test = {y_train.mean(), y_test.mean()}")

        X_train, X_test = impute_missing_numeric_discrete_data(X_train, X_test)

        X_train_features_added = model_input_features(X_train)
        X_test_features_added = model_input_features(X_test)

        print(f"top 1 record from train_data = {X_train_features_added.head(1)}")
        print(f"top 1 record from test_data = {X_test_features_added.head(1)}")

        return X_train_features_added, X_test_features_added, y_train, y_test

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_test_train_data()