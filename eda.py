# detect the types of columns in a data frame
# unique values, missing values
# minimum value, Q1, median, Q3, maximum, range, interquartile range
# like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# Most frequent values
# Histogram
# highly correlated variables, Spearman, Pearson and Kendall matrices
# Missing values matrix, count, heatmap, and dendrogram of missing values
# Text analysis learns about categories (Uppercase, Space), scripts (Latin, Cyrillic), and blocks (ASCII) of text data
# File and Image analysis extract file sizes, creation dates, and dimensions and scan for truncated images or those containing EXIF information


# remove duplicates
# drop irrelavant data/columns
# handle missing values
# handle outlier datapoints
# relationship analysis - correlation matrix

from bokeh.plotting import figure, output_file, show
from pathlib import Path
import bokeh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



df = pd.read_csv(Path("data-spotify/data.csv"))
df_artist = pd.read_csv(Path("data-spotify/data_by_artist.csv"))
df_genres_a = pd.read_csv(Path("data-spotify/data_by_genres.csv"))
df_genres_b = pd.read_csv(Path("data-spotify/data_w_genres.csv"))
df_year = pd.read_csv(Path("data-spotify/data_by_year.csv"))
df_super_genres = pd.read_json(Path("data-spotify/super_genres.json"))


#-------------------------------------------------------------------- Prilmnary Analysis
"""
Objective of the prilmnary analysis is to understand the
dataset with datatypes, missing values, duplicate entries,  
"""

df.drop_duplicates(inplace=True)
sampling_fromTop = df.head()
sampling_fromBottom = df.tail()
df.info()
df.isna().sum()

"""
[summary]
    total columns: 19
    total rows: 169909
    total numerical data columns: 15
    total categorical data columns: 4
    columns with null value present: 0
    total data in columns as 'na' - not applicable: 19
    *column-artists data is a string of list of strings 
    *column-release_date has only year others full date (YYYY-mm-dd)
    *colums with datetime was inported as numerical (year) and string (release_date) data type
"""
df_cont = df.select_dtypes(include=np.number)                        # dividing the dataset into continuous and categorical dataframes
df_cat = df.select_dtypes(include=object)


#-------------------------------------------------------------------- Univariate EDA
"""
To obtain information on values in a column w.r.t  weight, frequency and distribution

Categorical Data
    -useful charts: barchart
Continuous Data
    -useful charts: histograms, line chart, boxplot
"""

df_cont.describe()                                                   # descriptive statistics
df_cat.describe()

"""
[summary]
    values in column "duration_ms" can be converted to minutes, increases
    analysis efficiency without compromising the characteristics of data. 
"""
df_cont["duration_min"] = df_cont["duration_ms"] / 60000
df_cont.drop(["duration_ms"],axis=1, inplace=True)

def chart_histograms():
    df_cont.hist(figsize=(20, 20))
    plt.show()
chart_histograms()

#-------------------------------------------------------------------- Bivariate EDA
"""
To obtain information on relation between between two columns

Categorical to Categorical Relation
    -
Categorical to Continuous Relation
    -
Continuous to Continuous Relation
    -correlation
    -useful charts: heatmap, clustermap
"""
def chart_corr():
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")
    corr = df_cont.corr()
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    sns.clustermap(corr,cmap="coolwarm")
    plt.show()

"""
[summary]
"""


#-------------------------------------------------------------------- Multivariate EDA
"""
To obtain information on relation between between multiple columns/features

"""
def chart_featuresOverTimes():
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")
    columns = ["acousticness","danceability","energy","speechiness","liveness","valence"]
    for col in columns:
        x = df_cont.groupby("year")[col].mean()
        ax= sns.lineplot(x=x.index,y=x,label=col)
    ax.set_title('Audio characteristics over year')
    ax.set_ylabel('Measure')
    ax.set_xlabel('Year')
    plt.show()

"""
[summary]
"""





