#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


# In[25]:


def data(x):
    """Parameters:

    x (str): a string that represents the name of the CSV file to be read
    Returns:

    climatechange (pd.DataFrame): the pandas DataFrame that contains the data from the CSV file
    transpose (pd.DataFrame): the transposed pandas DataFrame """
    
    climatechange = pd.read_csv(x)
    
    # transpose the DataFrame
    transpose = climatechange.transpose()

    return climatechange, transpose


climatechange, _ = data('API_19_DS2_en_csv_v2_5346672.csv')

def clean(x):
    """
    Cleans the input DataFrame by filling any missing values with 0.

    Parameters:
    x (pandas DataFrame): the DataFrame to be cleaned

    Returns:
    Cleaned data """

    # count the number of missing values in each column of the DataFrame
    x.isnull().sum()
    # fill any missing values with 0 and update the DataFrame in place
    x.fillna(0, inplace=True)

    return

clean(climatechange)


def stats(a):
    """
    This function takes a pandas DataFrame `a` and performs several statistical calculations on the columns.
    It prints the summary statistics, correlation matrix, skewness, and kurtosis
    for the selected columns.  """

    # extract the columns from the 5th column onward and assign to variable "stats"
    stats = a.iloc[:, 4:]

    # calculate the skewness,kurtosis and Covariance
    print(skew(stats, axis=0, bias=True), kurtosis(
        stats, axis=0, bias=True), stats.describe(), stats.cov())


stats(climatechange)

def nitrous_bar(b):
    """
    This function takes a pandas DataFrame `b` containing data on worldbank climate change data and creates a bar
    plot of the percentage change in nitrous oxide emissions from 1990 for a selection of countries.
    """

    # Select rows where the "Indicator Name" column is "Nitrous oxide emissions (% change from 1990)"
    Nitrous = b[b['Indicator Name'] ==
                'Nitrous oxide emissions (% change from 1990)']

    # Select rows where the "Country Name" column is one of a list of countries
    Nitrous_emission = Nitrous[Nitrous['Country Name'].isin(['Nepal', 'China', 'India', 'New Zealand', 'Brazil',
                                                    'Dominican Republic', 'Pakistan', 'Iran, Islamic Rep.'])]


 # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(Nitrous_emission))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(15, 8))
    plt.bar(r1, Nitrous_emission['1995'], color='gold',
            width=bar_width, edgecolor='black', label='1991')
    plt.bar(r2, Nitrous_emission['1999'], color='orange',
            width=bar_width, edgecolor='black', label='1995')
    plt.bar(r3, Nitrous_emission['2000'], color='green',
            width=bar_width, edgecolor='black', label='2000')
    plt.bar(r4, Nitrous_emission['2003'], color='black',
            width=bar_width, edgecolor='black', label='2002')
    plt.bar(r5, Nitrous_emission['2005'], color='lightblue',
            width=bar_width, edgecolor='black', label='2007')
    plt.bar(r6, Nitrous_emission['2007'], color='lightblue',
            width=bar_width, edgecolor='black', label='2010')
    plt.bar(r7, Nitrous_emission['2009'], color='grey',
            width=bar_width, edgecolor='black', label='2012')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(Nitrous_emission))],
               Nitrous_emission['Country Name'])
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold')
    plt.ylabel('Nitrous_emission', fontweight='bold')
    plt.title('Nitrous oxide emissions (% change from 1990)', fontweight='bold')
    plt.legend()
    plt.show()


nitrous_bar(climatechange)

def fuel_emission(c):
    """
    This function takes a pandas DataFrame `c` containing data on worldbank climate change data and creates a bar
    plot of the percentage change in CO2 emissions from liquid fuel consumption for a selection of countries.
    """

    # Select rows where the "Indicator Name" column is "CO2 emissions from liquid fuel consumption (% of total)"
    fuel = c[c['Indicator Name'] ==
             'CO2 emissions from liquid fuel consumption (% of total)']

    # Select rows where the "Country Name" column is one of a list of countries
    fuel_emission = fuel[fuel['Country Name'].isin(['Nepal', 'China', 'India', 'New Zealand', 'Brazil',
                                                    'Dominican Republic', 'Pakistan', 'Iran, Islamic Rep.'])]
    # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(fuel_emission))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(15, 8))
    plt.bar(r1, fuel_emission['1995'], color='grey',
            width=bar_width, edgecolor='grey', label='1991')
    plt.bar(r2, fuel_emission['1999'], color='salmon',
            width=bar_width, edgecolor='black', label='1995')
    plt.bar(r3, fuel_emission['2000'], color='grey',
            width=bar_width, edgecolor='black', label='2000')
    plt.bar(r4, fuel_emission['2003'], color='darksalmon',
            width=bar_width, edgecolor='black', label='2002')
    plt.bar(r5, fuel_emission['2005'], color='coral',
            width=bar_width, edgecolor='black', label='2007')
    plt.bar(r6, fuel_emission['2007'], color='lightblue',
            width=bar_width, edgecolor='black', label='2010')
    plt.bar(r7, fuel_emission['2009'], color='sienna',
            width=bar_width, edgecolor='black', label='2012')
    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(fuel_emission))],
               fuel_emission['Country Name'])
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold')
    plt.ylabel('CO2 emissions from liquid fuel consumption (% of total)', fontweight='bold')
    plt.title(
        'CO2 emissions from liquid fuel consumption (% of total)', fontweight='bold')
    plt.legend()
    plt.show()

fuel_emission(climatechange)


def Energy_use_plot(c):
    """
    Plots a line graph of Energy use (%) of selected countries from the given dataframe

    Args:
    c: A Pandas dataframe containing the climate data

    """
    # filtering out the data related to Energy use for selected countries
    Energy_use = c[c['Indicator Name'] == 'Energy use (kg of oil equivalent per capita)']

    Energy= Energy_use[Energy_use['Country Name'].isin(['Nepal', 'China', 'India', 'New Zealand', 'Brazil',
                                                     'Dominican Republic', 'Pakistan', 'Iran, Islamic Rep.'])]

    # creating transpose of the filtered data
    Trans = Energy.transpose()
    Trans.rename(columns=Trans.iloc[0], inplace=True)
    Energy_transpose = Trans.iloc[4:]

    # Replacing the null values by zeros
    Energy_transpose.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(22, 10))
    plt.plot(Energy_transpose.index,
            Energy_transpose['Dominican Republic'], linestyle='dashed', label='Dominican Republic')
    plt.plot(Energy_transpose.index,
             Energy_transpose['China'], linestyle='dashed', label='China')
    plt.plot(Energy_transpose.index,
             Energy_transpose['Brazil'], linestyle='dashed', label='Brazil')
    plt.plot(Energy_transpose.index,
             Energy_transpose['India'], linestyle='dashed', label='India')
    plt.plot(Energy_transpose.index,
             Energy_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(Energy_transpose.index,
             Energy_transpose['Nepal'], linestyle='dashed', label='Nepal')
    plt.plot(Energy_transpose.index,
             Energy_transpose['Pakistan'], linestyle='dashed', label='Pakistan')
    plt.plot(Energy_transpose.index,
             Energy_transpose['Iran, Islamic Rep.'], linestyle='dashed', label='Iran, Islamic Rep.')
    # Setting x limit
    plt.xlim('2000', '2012')
    # Adding labels to the axis
    plt.xlabel('Year', fontsize=15, fontweight='bold')
    plt.ylabel('percentage of Energy use', fontsize=15, fontweight='bold')
    plt.title('Energy use (kg of oil equivalent per capita)', fontsize=15, fontweight='bold')
    plt.legend()
    plt.show()

Energy_use_plot(climatechange)


def Renewable_electricity_output_plot(c):
    """
    This function takes a pandas dataframe 'c' as input and plots a line graph showing the renewable electricity output
    annual percentage for 10 different countries.

    Parameters:
    c (pd.DataFrame): pandas dataframe """

    # filter the rows which have indicator name as 'Renewable electricity output (% of total electricity output)'
    renewable_electricity = c[c['Indicator Name'] ==
                              'Renewable electricity output (% of total electricity output)']

    # filter the rows for the 10 selected countries
    renewable = renewable_electricity[renewable_electricity['Country Name'].isin(['Nepal', 'China', 'India', 'New Zealand', 'Brazil',
                                                            'Dominican Republic', 'Pakistan', 'Iran, Islamic Rep.'])]

    # # creating transpose of the filtered data
    Tran = renewable.transpose()
    Tran.rename(columns=Tran.iloc[0], inplace=True)
    renewable_transpose = Tran.iloc[4:]
    # Replacing the null values by zeros
    renewable_transpose.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(22, 10))
    plt.plot(renewable_transpose.index,
             renewable_transpose['Dominican Republic'], linestyle='dashed', label='Dominican Republic')
    plt.plot(renewable_transpose.index,
             renewable_transpose['China'], linestyle='dashed', label='China')
    plt.plot(renewable_transpose.index,
             renewable_transpose['Brazil'], linestyle='dashed', label='Brazil')
    plt.plot(renewable_transpose.index,
             renewable_transpose['India'], linestyle='dashed', label='India')
    plt.plot(renewable_transpose.index,
             renewable_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(renewable_transpose.index,
             renewable_transpose['Nepal'], linestyle='None', label='Nepal')
    plt.plot(renewable_transpose.index,
             renewable_transpose['Pakistan'], linestyle='dashed', label='Pakistan')
    plt.plot(renewable_transpose.index,
             renewable_transpose['Iran, Islamic Rep.'], linestyle='dashed', label='Iran, Islamic Rep.')
    # setting x limit
    plt.xlim('2001', '2012')
    # adding labels to the axis
    plt.xlabel('Year', fontsize=15, fontweight='bold')
    plt.ylabel('Renewable electricity output', fontsize=15, fontweight='bold')
    plt.title('Renewable electricity output (% of total electricity output)',
              fontsize=15, fontweight='bold')
    plt.legend()
    plt.show()

Renewable_electricity_output_plot(climatechange)

def heatmap_Pak(x):
    """
    A function that creates a heatmap of the correlation matrix between different indicators for Pakistan.

    Args:
    x (pandas.DataFrame): A DataFrame containing data on different indicators for various countries.

    Returns:
    This function plots the heatmap ."""

    # Specify the indicators to be used in the heatmap
    indicator = ['Nitrous oxide emissions (% change from 1990)',
                 'CO2 emissions from solid fuel consumption (kt)',
                 'Energy use (kg of oil equivalent per capita)',
                 'Renewable electricity output (% of total electricity output)',
                 'Forest area (% of land area)']

    # Filter the data to keep only China's data and the specified indicators
    Pak = x.loc[x['Country Name'] == 'Pakistan']
    Pakistan = Pak[Pak['Indicator Name'].isin(indicator)]
    # Pivot the data to create a DataFrame with each indicator as a column
    Pakistan_df = Pakistan.pivot_table(Pakistan, columns=x['Indicator Name'])
    # Compute the correlation matrix for the DataFrame
    Pakistan_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(Pakistan_df.corr(), fmt='.2g', annot=True,
                cmap='magma', linecolor='black')
    plt.title('Pakistan', fontsize=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


heatmap_Pak(climatechange)



def heatmap_Dma(x):
    """
    A function that creates a heatmap of the correlation matrix between different indicators for Dominican_Republic.

    Args:
    x (pandas.DataFrame): A DataFrame containing data on different indicators for various countries.

    Returns:
    This function plots the heatmap.
    """

    # Specify the indicators to be used in the heatmap
    indicator = ['Nitrous oxide emissions (% change from 1990)',
                 'CO2 emissions from solid fuel consumption (kt)',
                 'Energy use (kg of oil equivalent per capita)',
                 'Renewable electricity output (% of total electricity output)',
                 'Forest area (% of land area)']

    # Filter the data to keep only Dominican Republic's data and the specified indicators
    Dma = x.loc[x['Country Name'] == 'Dominican Republic']
    Dominican_Republic = Dma[Dma['Indicator Name'].isin(indicator)]

    # Pivot the data to create a DataFrame with each indicator as a column
    Dominican_Republic_df = Dominican_Republic.pivot_table(
        Dominican_Republic, columns=x['Indicator Name'])
    # Compute the correlation matrix for the DataFrame
    Dominican_Republic_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(Dominican_Republic_df.corr(), fmt='.2g',
                annot=True, cmap='magma', linecolor='black')
    plt.title('Dominican Republic', fontsize=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

heatmap_Dma(climatechange)


# In[ ]:





# In[ ]:





# In[ ]:




