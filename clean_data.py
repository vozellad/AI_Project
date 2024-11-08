import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    # Clean the column names (specifically 'Daytime/evening attendance')
    df.columns = df.columns.str.strip().str.replace('\t', '')
    df.columns = df.columns.str.replace('"', '')

    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    # Change classified text under ‘Target’ column to be integers.
    df['Target'] = LabelEncoder().fit_transform(df['Target'])

    # Strip leading and trailing white-spaces.
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop empty rows.
    df.dropna(how='all', inplace=True)

    # Replace missing values with the mean.
    df.fillna(df.mean(), inplace=True)

    # Set proper precision.
    df = df.round(3)

    return df


def visualize_data(df):
    df['Target'].value_counts().plot(kind='pie', autopct='%1.3f%%')
    plt.title('Pie chart of Target')
    plt.show()


def iqr_processing(df):
    variance = df.var()
    std_dev = df.std()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    print('Variance:')
    print(variance, '\n')
    print('Standard Deviation:')
    print(std_dev, '\n')
    print('Inter-quartile Range (IQR):')
    print(iqr, '\n')

    print('\n\n')

    outliers = df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))]
    print("Outliers:")
    print(outliers)
    outliers.to_csv("outliers.xlsx", index=False)



def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename, delimiter=';')
    clean_data(df)
    df.to_csv("cleaned_data.xlsx", index=False)

    iqr_processing(df)
    visualize_data(df)
    print(df.describe())


if __name__ == '__main__':
    main()
