import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def generate_categorical_data_plot(df):
    #Missing Values
    missing_ailment = df['Ailment'].isnull().sum()
    print(f"Missing Ailment values: {missing_ailment}")

    #Pie Chart
    plt.figure(figsize=(10, 10))
    df['Ailment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                        labeldistance=1.1, pctdistance=0.85)
    plt.title('Proportion of Each Ailment')
    plt.ylabel('')
    plt.show()
    
    #Missing Values
    missing_ailment = df['Ailment'].isnull().sum()
    print(f"Missing Ailment values: {missing_ailment}")
    print("---" * 50)
    # Cleaned up Dataframe
    print("--------Cleaned up dataframe-------------")
    print(df.head())
    
def generate_histogram_kde_plots(df):
    """
    7. Create histograms and KDE plots (find out what KDE plots represent) for all the columns and analyze
    them. Any significant conclusions?
    """
    fig, axes = plt.subplots(6, 4, figsize=(10, 10))
    axes = axes.flatten()
    num_columns = df.select_dtypes(include=['number']).columns

    for i, col in enumerate(num_columns):
        sns.histplot(df[col], ax=axes[i], kde=True, color=sns.color_palette('Set2')[1], bins=10)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')

    plt.suptitle('Histogram and KDE plots of all numeric columns', fontsize=15)
    plt.tight_layout()
    plt.show()
    """
    Observations :

    1. The dataset shows mixed distribution patterns — some normal-like, others multimodal or skewed.
    2. Multiple peaks in several columns may indicate underlying groups or categories
    3. Few variables (like p04, p05) dominate in scale → scaling/normalization will be essential before modeling.
    4. No column shows extreme outlier clusters, indicating reasonably clean numeric data.
    """

def generate_box_plots(df):
    """
    8. Create box plots for all the columns, individually, and analyze them. What are your conclusions?
    """

    fig, axes = plt.subplots(6, 4, figsize=(10, 15))
    axes = axes.flatten()
    num_columns = df.select_dtypes(include=['number']).columns

    for i, col in enumerate(num_columns):
        sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette('Set3')[4], orient='v', showmeans=True)
        axes[i].set_title(col, fontsize=5)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')

    plt.suptitle('Box plots of all numeric columns', fontsize=10)
    plt.tight_layout()
    plt.show()

    """
    Observations :
    1. p01, p02, p05, p09, p10, p13, p17, p19, p20 — mean slightly above median → mild right skew.
    2. p03, p04, p14, p15 — mean below median → mild left skew.
    3. The rest look approximately symmetric.
    4. No extreme outliers are visible — whiskers are even and smooth. Data seems fairly clean
    5. Some columns (like p04, p05, p13) have much larger numerical ranges than others (e.g., p22, p23).
    """
    
def generate_box_plot_for_all_at_once(df):
    """
    9. Create box plots for all the numerical columns on a common scale, and in a single plot, and analyze this plot. What do you observe? What are its implications?
    """

    numeric_df = df.select_dtypes(include='number')

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=numeric_df, showmeans=True, color='SkyBlue')

    plt.title('Box plots of all numeric columns (on common scale)', fontsize=14)
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

    """
    Observations:
    1 . Differences in Scale
        Columns like p04 and p05 have very high numeric ranges (hundreds of thousands or tens of thousands).
        Others like p06, p16, p22, p23 are small-valued (e.g., in single digits or decimal range).

    2. There are no outliers in any of the columns

    3. We need to standardization or normalization before creating the model
    """

def generate_pairwise_scatter_plot(df):
    """
    10. To understand the relationships between the columns, create the following:
        a. Pairwise scatter plots
        b. Heatmap of pairwise correlation coefficients
        c. What are your conclusions based on these plots?
    """

    # a. Pairwise scatter plots
    num_columns = df_clean.select_dtypes(include=['number']).columns
    sns.pairplot(df_clean, vars=num_columns[:4], hue='Ailment', diag_kind='kde')
    plt.suptitle("Pairwise Scatter Plots for Selected Variables", y=1.02)
    plt.show()

    # b. Heatmap
    corr = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Heatmap of Pairwise Correlation Coefficients', fontsize=10)
    plt.show()

    #c. Conclusion
    """
    Features are independent of each other . We need to retain all features as each my add unique information
    """
def generate_scatter_plots(df):
    plt.figure(figsize=(10,20))
    number_cols = df.select_dtypes(include=['number']).columns
    subplot_count = 1
    for i, col in enumerate(number_cols):
        plt.subplot(6, 4, subplot_count)
        plt.scatter(df.index, df[col])
        plt.xlabel('index')
        plt.ylabel('values')
        plt.tight_layout()
        subplot_count+=1
    plt.show()
    

def get_metadata_details(df):
    row, column = df.shape
    print(f"Rows: {row} Columns: {column}")
    null_value_in_columns = df.isna().any().sum()
    null_value_in_rows = df.isna().any(axis=1).sum()
    print(f"Null value in Rows: {null_value_in_rows}  & Null value in Columns: {null_value_in_columns}")
    
def generate_analysis_and_conclusions(df):
    """
    3.  Create appropriate Descriptive Statistics information for every column, based on it’s level of
        measurement, and organize all this data in Tables for easy analysis and decision making.
            a. Analyze the descriptive statistics thus generated and make your initial conclusions
    """
    numeric_df = df.select_dtypes(include=['number'])
    desc = numeric_df.describe().T

    desc['skewness'] = numeric_df.skew()
    desc['kurtosis'] = numeric_df.kurtosis()

    desc = desc.round(5)
    print("-" * 50)
    print("------ DESCRIPTION OF Q3 -------")
    print(desc)
    print("------ DESCRIPTION OF Q3 -------")
    """
    Observations :
        a. Some variables (like p04, p05) have large ranges, suggesting these might represent quantitative lab measures with wide patient variability.

        b. Others (like p06, p22, p23) show small ranges, likely indicating standardized or ratio-based metrics.

        2. Outliers and distribution:

            a. A few features (p04, p05, p17) show large standard deviations relative to their means, indicating possible outliers or heterogeneity among patient groups.
    """
def generate_logistic_regression(df):
    """
    12. Create a Logistic Regression model using this data. Create the train/test metrics. Hope you have not forgotten to split the data into train and test data!
    """
    x = df.drop('Ailment', axis=1)
    y = df['Ailment']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print('\nConfusion Metrics\n', confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def generate_common_scale_box_plot(df_clean):
    """
    13. Normalize the numerical columns and re-create the common-scale Box-plot. Interpret the results
    """
    numeric_cols = df_clean.select_dtypes(include='number').columns
    df_normalized = df_clean.copy()
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_normalized, showmeans=True, color='SkyBlue')

    plt.title('Box plots of all numeric columns NORMALIZED (on common scale)', fontsize=14)
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    """
    Observation:
    1. We do not see the difference in scale between all columns. All column values are between now -1 and 1 with mean 0
    """   
    
def generate_regression_on_normaliezed_data(df):
    """
    14. Create a Logistic Regression model using the normalized data. Create the train/test metrics and compare with the metrics of step '12'. What are your observations?
    """
    x = df_clean.drop('Ailment', axis=1)
    y = df_clean['Ailment']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print('\nConfusion Metrics\n', confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    """
    Observation:

    1. Accuracy of the model increased when data is normalized
    """

if __name__ == "__main__":
    df = pd.read_csv('patient-data.csv')
    print("✅ Data read from csv")
    #Q1
    get_metadata_details(df)
    #Q2
    print(df.info())
    #Q3
    generate_analysis_and_conclusions(df)
    #Q4 Scatterplots of all columns
    generate_scatter_plots(df)
    """
    5.Based on the information available so far, what will be your strategy for dealing with the missing values?
        Make some initial decisions at this stage.
      - We can remove the missing values as it contributes only 0.17% of total data
      - You don’t want to make assumptions by imputing
    """
    df_clean = df.dropna()
    print(df_clean.head())
    #Q6 Analyze the ailment column to draw conclusion
    generate_categorical_data_plot(df_clean)
    #Q7
    generate_histogram_kde_plots(df_clean)
    #Q8
    generate_box_plots(df_clean)
    #Q9
    generate_box_plot_for_all_at_once(df_clean)
    #Q10
    generate_pairwise_scatter_plot(df_clean)
    """ Q11. Based on all the analysis so far, what is your final conclusion regarding the handling of missing values?
    Implement your decision!**

    We have decided to remove all rows having missing values as they constitute  < 1% of the total data, so can be safely removed
    df_clean contains the dataframe after dropping all missing values
    """
    #Q12
    generate_logistic_regression(df_clean)
    #Q13
    generate_common_scale_box_plot(df_clean)
    #Q14
    generate_regression_on_normaliezed_data(df_clean)
    """
    BTW, in steps 12 and 14 did you pass the Ailments data as is – that is, as text information? Did the Python functions
    work? Why?

    - Scikit-learns LogisticRegression automatically encodes string labels internally using Label Encoding.
    - You dont need to manually convert them
        model.fit(X_train, y_train) will handle string targets automatically.
    - But this works only for the target column, not for features.
    """
    
    
    