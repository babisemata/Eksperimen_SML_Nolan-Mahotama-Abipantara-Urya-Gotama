import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(path):
    return pd.read_csv(path)


def drop_unused_columns(df):
    cols_to_drop = [
        'EmployeeNumber',
        'EmployeeCount',
        'Over18',
        'StandardHours'
    ]
    return df.drop(columns=cols_to_drop)




def binning_age(df):
    age_bins = [17, 25, 35, 45, 60]
    age_labels = ['Early Career', 'Mid Career', 'Senior', 'Late Career']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    return df

def encode_target(df):
    df['Attrition'] = df['Attrition'].map({
        'Yes': 1,
        'No': 0
    })
    return df

def encode_features(df):
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    binary_mapping = {
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0
    }

    for col in ['OverTime', 'Gender']:
        X[col] = X[col].map(binary_mapping)

    nominal_cols = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'JobRole',
        'MaritalStatus',
        'AgeGroup'
    ]

    X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test):
    output_dir = "HR-Employee-Attrition_preprosessing"
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Attrition'] = y_train.reset_index(drop=True)

    test_df = X_test.copy()
    test_df['Attrition'] = y_test.reset_index(drop=True)

    train_df.to_csv(f"{output_dir}/train_processed.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_processed.csv", index=False)


def main():
    input_path = "HR-Employee-Attrition_raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    df = load_data(input_path)
    df = drop_unused_columns(df)
    df = encode_target(df)
    df = binning_age(df)

    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    save_processed_data(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
