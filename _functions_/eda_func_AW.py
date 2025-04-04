#!/usr/bin/env python
# coding: utf-8

# In[5]:


import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


py.init_notebook_mode(connected=True)
sns.set(style="white", color_codes=True)


def get_column_names(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    """
    Extracts the names of numerical and categorical columns from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: Two lists:
            - num_vars (list): Names of numerical columns (int or float types).
            - cat_vars (list): Names of categorical columns (object or category types).
    """
    num_var = df.select_dtypes(include=['int', 'float']).columns
    print()
    print('Numerical variables are:\n', num_var)
    print('-------------------------------------------------')

    categ_var = df.select_dtypes(include=['category', 'object']).columns
    print('Categorical variables are:\n', categ_var)
    print('-------------------------------------------------')
    return num_var, categ_var


def percentage_nullValues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the percentage of missing (NaN) values in each column of a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with one column ('Percentage_NaN'), 
                    sorted in descending order of missing value percentage.
    """
    null_perc = round(df.isnull().sum() / df.shape[0], 3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc = null_perc.sort_values(by=['Percentage_NaN'], ascending=False)
    return null_perc


# In[26]:


def select_threshold(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """
    Filters out columns that exceed a specified threshold of missing values.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        thr (float): Maximum acceptable percentage of missing values.

    Returns:
        pd.DataFrame: A DataFrame containing only columns with NaN percentage less than the threshold.
    """
    null_perc = percentage_nullValues(df)

    col_keep = null_perc[null_perc['Percentage_NaN'] < thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:', len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c = df[col_keep]

    return data_c


# In[33]:


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in a DataFrame:
    - Numerical columns are filled with the mean.
    - Categorical columns are filled with the mode.

    Parameters:
        data (pd.DataFrame): The DataFrame with missing values.

    Returns:
        pd.DataFrame: The DataFrame with NaN values filled.
    """
    for column in df:
        if df[column].dtype != 'object':
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    print('Number of missing values on your dataset are')
    print()
    print(df.isnull().sum())
    return df


# In[2]:

def outliers_box(df: pd.DataFrame, name_of_feat: str) -> None:
    """
    Generates two interactive Plotly box plots for a numeric feature:
    1. One showing all data points.
    2. One showing only suspected outliers.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        nameOfFeature (str): The column name for which to plot outliers.
    """
    trace0 = go.Box(
        y=df[name_of_feat],
        name="All Points",
        jitter=0.3,
        pointpos=-1.8,
        boxpoints='all',  # define that we want to plot all points
        marker=dict(
            color='rgb(7,40,89)'),
        line=dict(
            color='rgb(7,40,89)')
    )

    trace1 = go.Box(
        y=df[name_of_feat],
        name="Suspected Outliers",
        boxpoints='suspectedoutliers',  # define the suspected Outliers
        marker=dict(
            color='rgba(219, 64, 82, 0.6)',
            # outliercolor = 'rgba(219, 64, 82, 0.6)',
            line=dict(
                outlierwidth=2)),
        line=dict(
            color='rgb(8,81,156)')
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title="{} Outliers".format(name_of_feat)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    # fig.write_html("{}_file.html".format(nameOfFeature))

# In[3]:


def corr_coef(df: pd.DataFrame) -> None:
    """
    Generates a correlation matrix heatmap for numerical features in a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the heatmap using seaborn and matplotlib.
    """
    num_vars, categ_var = get_column_names(df)
    data_num = df[num_var]
    data_corr = data_num.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_corr,
                xticklabels=data_corr.columns.values,
                yticklabels=data_corr.columns.values,
                annot=True, vmax=1, vmin=-1, center=0, cmap=sns.color_palette("RdBu_r", 7))


# In[4]:

def corr_coef_threshold(df: pd.DataFrame) -> None:
    """
    Displays a triangular correlation heatmap (upper triangle masked) for a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the heatmap.
    """
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Draw the heatmap
    sns.heatmap(df.corr(), annot=True, mask=mask, vmax=1, vmin=-1,
                cmap=sns.color_palette("RdBu_r", 7))


def outlier_treatment(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Removes outliers from a numerical column based on the interquartile range (IQR) method.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        colname (str): Name of the column to process.

    Returns:
        pd.DataFrame: DataFrame with outliers removed for the specified column.
    """
    # Calculate the percentiles and the IQR
    Q1, Q3 = np.percentile(df[col_name], [25, 75])
    IQR = Q3 - Q1

    # Calculate the upper and lower limit
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    # Drop the suspected outliers
    df_clean = df[(df[col_name] > lower_limit) & (df[col_name] < upper_limit)]

    print('Shape of the raw data:', df.shape)
    print('..................')
    print('Shape of the cleaned data:', df_clean.shape)
    return df_clean


def outliers_loop(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies outlier treatment to all columns in a numerical DataFrame using IQR method.

    Parameters:
        numerical_df (pd.DataFrame): DataFrame containing only numerical columns.

    Returns:
        pd.DataFrame: A cleaned DataFrame with outliers removed from all columns.
    """
    for item in np.arange(0, len(numerical_df.columns)):
        if item == 0:
            df_c = outlier_treatment(numerical_df, numerical_df.columns[item])
        else:
            df_c = outlier_treatment(df_c, numerical_df.columns[item])
    return df_c
