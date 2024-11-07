import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


def clean_data(df):
    df.drop(["prodt", "score_plus_1"], axis=1, inplace=True)

    df.drop(df[df['pstl_yr'] < 2023].index, inplace=True)

    columns_to_check = ['orgn_area', 'orgn_dist', 'destn_area', 'destn_dist']
    # drop rows with -1 value
    condition = df[columns_to_check].isin([-1]).any(axis=1)
    df.drop(df[condition].index, inplace=True)

    df.drop_duplicates(inplace=True)

    df['avg_days_to_delr'] = df['avg_days_to_delr'].round(10)
    df['score'] = df['score'].round(10)

    df.dropna(subset=['avg_days_to_delr', 'score'], inplace=True)

    df.dropna(how='all', inplace=True)


def visualize_data(df, column_title):
    # shape
    plt.figure(figsize=(12, 5))

    # histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[column_title], bins=100, kde=True)
    plt.ylabel('Frequency')

    # boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column_title])

    plt.tight_layout()
    plt.show()


def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename)

    output_filename = "cleaned_data.xlsx"
    df.to_csv(output_filename, index=False)

    clean_data(df)

    x = df['']
    y = df['avg_days_to_delr']

    visualize_data(df, 'score')
    visualize_data(df, 'avg_days_to_delr')

    print(df.describe())


if __name__ == '__main__':
    main()
