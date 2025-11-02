import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    

if __name__ == "__main__":
    df = pd.read_csv('patient-data.csv')
    print("✅ Data read from csv")
    #Q1
    get_metadata_details(df)
    #Q2
    print(df.info())
    #Q4 Scatterplots of all columns
    generate_scatter_plots(df)
    df_clean = df.dropna()
    print(df_clean.head())
    #Q6 Analyze the ailment column to draw conclusion
    generate_categorical_data_plot(df)
    
    
    
    