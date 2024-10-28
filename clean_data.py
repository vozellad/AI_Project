import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_general_characteristics(column_title):
    # center
    mean = df[column_title].mean()
    median = df[column_title].median()
    print("Centers:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")

    # spread
    variance = df[column_title].var()
    std_dev = df[column_title].std()
    iqr = np.percentile(df[column_title], 75) - np.percentile(df[column_title], 25)
    print("\nMeasures of Spread:")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")
    print(f"IQR: {iqr}")

    # modality
    mode = df[column_title].mode().values
    print(f"Mode: {mode}")

    # outliers
    q = np.percentile(df[column_title], 25)
    q3 = np.percentile(df[column_title], 75)
    iqr = q3 - q
    lower_bound = q - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column_title] < lower_bound) | (df[column_title] > upper_bound)]
    print(outliers if not outliers.empty else None)

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


filename = "extract0"  # no file extension because I downloaded it on Linux. It's a CSV file.
df = pd.read_csv(filename)

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

output_filename = "cleaned_data.xlsx"
df.to_csv(output_filename, index=False)

# show_general_characteristics('score')
# show_general_characteristics('avg_days_to_delr')

print(df.describe())