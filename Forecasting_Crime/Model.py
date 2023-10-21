import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet


def load_csv(file):
    return pd.read_csv(file, error_bad_lines = False)

def main():
    chicago_df_1 = load_csv("Chicago_Crimes_2001_to_2004.csv")
    chicago_df_2 = load_csv("Chicago_Crimes_2005_to_2007.csv")
    chicago_df_3 = load_csv("Chicago_Crimes_2008_to_2011.csv")
    chicago_df_4 = load_csv("Chicago_Crimes_2012_to_2017.csv")
    chicago_df_total = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3, chicago_df_4])

    chicago_df_total.drop(['Unnamed: 0', 'Case Number', 'ID', 'IUCR', 'X Coordinate',
    'Y Coordinate', 'Updated On', 'Year', 'FBI Code', 'Beat', 'Ward', 
    'Community Area', 'Location', 'District', 'Latitude', 'Longitude'], inplace=True, axis=1)

    chicago_df_total.Date = pd.to_datetime(chicago_df_total.Date, format= '%m/%d/%Y %I:%M:%S %p')
    #Converts the date into the indexes
    chicago_df_total.index = pd.DatetimeIndex(chicago_df_total.Date)

    chicago_df_total['Primary Type'].value_counts()
    order_data = chicago_df_total['Primary Type'].value_counts().iloc[:15].index

    # plt.figure(figsize = (15,10))
    # sns.countplot(y = 'Primary Type', data = chicago_df_total, order = order_data)

    # plt.figure(figsize=(15,10))
    # sns.countplot(y='Location Description', data=chicago_df_total, order=chicago_df_total['Location Description'].value_counts().iloc[:15].index)
    # plt.show()

    #Gives us the number of crimes per year
    # plt.plot(chicago_df_total.resample('Y').size())
    # plt.title('Crime Count Per Year')
    # plt.xlabel('Years')
    # plt.ylabel('Number of Crimes')

    # #Gives us the number of crimes per month 
    # plt.plot(chicago_df_total.resample('M').size())
    # plt.title('Crime Count Per Month')
    # plt.xlabel('Months')
    # plt.ylabel('Number of Crimes')

    # #Gives us the number of crimes per quarter
    # plt.plot(chicago_df_total.resample('Q').size())
    # plt.title('Crime Count Per Quarter')
    # plt.xlabel('Quarter')
    # plt.ylabel('Number of Crimes')
    # plt.show()

    #We transform the data so that we have the month as date 
    #and in another column we have the number of crimes of that month
    chicago_prophet = chicago_df_total.resample('M').size().reset_index()

    #Rename columns
    chicago_prophet.columns = ['Date', 'Crime Count']

    #We change the name again of the columns so that we can use facebook prophet
    chicago_prophet_final = chicago_prophet.rename(columns = {'Date': 'ds', 'Crime Count': 'y'})

    #We assign our data to our "prophet"
    model = Prophet()
    model.fit(chicago_prophet_final)
    #Start predicting :)
    future = model.make_future_dataframe(periods = 365)
    forecast = model.predict(future)
    #print(forecast)
    model.plot(forecast, xlabel='Date', ylabel='Crime Rate')
    #with .plot_components(forecast) we can predict the seasonal trend
    plt.show()


if '__main__' == __name__:
    main()