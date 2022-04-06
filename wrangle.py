import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from env import username, password, host


######## acquire ##########

def get_zillow_data(use_cache=True):
    '''This function returns the data from the zillow database in Codeup Data Science Database. 
    In my SQL query I have joined all necessary tables together, so that the resulting dataframe contains all the 
    information that is needed
    '''
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/'
    query = '''
    select * from properties_2017
    join predictions_2017 using(parcelid)
    left join airconditioningtype using(airconditioningtypeid)
    left join architecturalstyletype using(architecturalstyletypeid)
    left join buildingclasstype using(buildingclasstypeid)
    left join heatingorsystemtype using(heatingorsystemtypeid)
    left join propertylandusetype using(propertylandusetypeid)
    left join storytype using(storytypeid)
    left join typeconstructiontype using(typeconstructiontypeid)
    where latitude IS NOT NULL
    and longitude IS NOT NULL
    '''
    df = pd.read_sql(query, database_url_base + 'zillow')
    df.to_csv('zillow.csv', index=False)
    return df


######## cleaning/prep ###############

def clean_zillow(df):
    '''takes in zillow and removes redundant columns, and returns a clean version '''
    # selecting landusetypeid for single family homes
    df = df[df.propertylandusetypeid.isin([260,261,262,263,264,265,266,268,273,275,276,279])]
    
    # set datafram to houses with at least 1 bed/bath each
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)] 

    # filling null values with most popular
    df.unitcnt = df.unitcnt.fillna(1.0)

    # most properties in southern california don't have AC
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('None')  

    # handle missing values
    df = handle_missing_values(df)

    # add column with information from fips column
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))  

    # add a column with information from yearbuilt column
    df['age'] = 2022 - df.yearbuilt

    # columns to drop
    remove_columns = ['propertylandusetypeid', 'calculatedbathnbr', 'heatingorsystemtypeid', 'parcelid', 'propertyzoningdesc', 'id', 'id.1', 'rawcensustractandblock', 'fips', 'yearbuilt']
    df = df.drop(columns = remove_columns)
    
    col_list = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount', 'logerror'] 
    df = remove_outliers(df, 3.0, col_list)

    return df

def handle_missing_values(df, prop_required_column = .55, prop_required_row = .7):
    ''' takes in a datafram and is defaulted to have at least 55 percent of values for columns and 70 percent for rows'''
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df

def nulls_by_col(df):
    ''' Takes in a dataframe and will output a dataframe of information for columns missing'''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows':num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing

def nulls_by_row(df):
    ''' Takes in a dataframe and will output a dataframe of information for rows missing'''
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing':num_cols_missing, 'pct_cols_missing':pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing


def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a dataframe, value of k, and a list of columns, removes outliers greater than k times IQR above the 75th percentile 
    and lower than k times IQR below the 25th percentile from the list of columns specified and returns a dataframe. 
    k = 1.5 is mild outliers left
    k = 3.0 is median outliers left
    '''
    # loop through each column
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles    
        iqr = q3 - q1   # calculate interquartile range    
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
 
def split_data(df, random_state=123, stratify=None):
    '''
    This function takes in a dataframe and splits the data into train, validate and test samples. 
    Test, validate, and train are 20%, 24%, & 56% of the original dataset, respectively. 
    The function returns train, validate and test dataframes.
    '''
   
    if stratify == None:
        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state)

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, random_state=random_state)
    else:
        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state, stratify=df[stratify])

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, 
                            random_state=random_state,stratify=train_validate[stratify])

    # results in 3 dataframes
    return train, validate, test

def handle_nulls(train, validate, test):
    # continuous values filled with mode
    cols = [
    'buildingqualitytypeid',
    'regionidzip',
    'regionidcity',
    'censustractandblock',
    ]

    for col in cols:
        mode = int(train[col].mode())
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value =mode, inplace = True)
        test[col].fillna(value=mode, inplace = True)

    # categorical columns filled with median

    cols = [
    'buildingqualitytypeid',
    'taxamount',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'structuretaxvaluedollarcnt',
    'finishedsquarefeet12',
    'calculatedfinishedsquarefeet',
    'fullbathcnt',
    'lotsizesquarefeet'
    ]

    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace = True)
        validate[col].fillna(median, inplace = True)
        test[col].fillna(median, inplace = True)

    return train, validate, test