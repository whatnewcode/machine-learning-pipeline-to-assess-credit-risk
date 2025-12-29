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

    # split input data into train and test
    test_data, train_data = train_test_split(loan_data_df, test_size=0.8, train_size=0.2, random_state=1)

    print(f"train_data = {len(train_data)}")
    print(f"test_data = {len(test_data)}")
    print(f"top 1 record from train_data = {train_data.head(1)}")
    print(f"top 1 record from test_data = {test_data.head(1)}")

except Exception as e:
    print(f"Error: {e}")