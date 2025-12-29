import pandas as pd
from sklearn.model_selection import train_test_split

try:
    # load loan_data.csv into panda DF
    try:
        loan_data_df = pd.read_csv("loan_data.csv")
        print(f"total rows from load_data df = {len(loan_data_df)} ")
        # print(f"top 5 rows = {loan_data_df.head(5)}")
    except Exception as e:
        print(f"Error while loading loan_data.csv into Pandas Data Frame - {e}")

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

except Exception as e:
    print(f"Error: {e}")