from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv(Path("data-spotify/data.csv"))
#--------------------------------------------------------------------------------------- Prilmnary Analysis
"""
Objective of the prilmnary analysis is to understand the
dataset w.r.t datatypes and missing values 
"""

df.drop_duplicates(inplace=True)
sampling_fromTop = df.head()
sampling_fromBottom = df.tail()
df.info()
df.isna().sum()

"""
[summary]
    total features: 19
    total datapoints: 169909
    the dataset has  15 - continous and 4 - catogorecal features
    the dataset has zero null and na values accross features
    "artists" data is a string of list of strings 
    "release_date" has mixed datapoints: year (YYYY) and date (YYYY-mm-dd), requires processing
    colums with datetime were imported as numerical ("year") and string ("release_date") data type
"""
df_cont = df.select_dtypes(include=np.number)                        # dataframe of continuous data
df_cat = df.select_dtypes(include=object)                            # dataframe of categorical data


#--------------------------------------------------------------------------------------- Univariate EDA
"""
To obtain information on values of a feature w.r.t weight, frequency and distribution

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


#--------------------------------------------------------------------------------------- Bivariate EDA
"""
To obtain information on relation between between two features

Categorical to Categorical Relation
    -
Categorical to Continuous Relation
    -
Continuous to Continuous Relation
    -correlation
    -useful charts: heatmap, clustermap
"""
def chart_corr():                                                    # correlation between features
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")
    corr = df_cont.corr()
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    sns.clustermap(corr,cmap="coolwarm")
    plt.show()

"""
[summary]
"""


#--------------------------------------------------------------------------------------- Multivariate EDA
"""
To obtain information on relation between between multiple features

"""
def chart_featuresOverTimes():                                       # time series visualizations 
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



# rate of songs added in spotify over the years
# feature trend analysis over the decades



