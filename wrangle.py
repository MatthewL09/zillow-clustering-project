import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import os
from env import username, password, host


######  acquire  ######

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
    '''takes in zillow and removes redundant columns, and returns a clean version
     '''
     # selecting landusetypeid for single family homes
    df = df[df.propertylandusetypeid.isin([260,261,262,263,264,265,266,268,273,275,276,279])]

    # set dataframe to houses with at least an acceptable amount of bedroom/baths, sqft is minimum requirement for LA, yearbuilt LA aquaduct completion in 1913
    df = df[(df.bedroomcnt < 8) & ( df.bathroomcnt < 7) & (df.yearbuilt >= 1920) & (df.calculatedfinishedsquarefeet > 800) ] 

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

    # columns to drop
    remove_columns = ['propertylandusetypeid', 'calculatedbathnbr', 'heatingorsystemtypeid', 'parcelid', 'propertyzoningdesc', 'id', 'id.1', 'rawcensustractandblock',
     'fips',  'fullbathcnt',  'buildingqualitytypeid', 'roomcnt', 'assessmentyear', 'finishedsquarefeet12', 'propertycountylandusecode']
    df = df.drop(columns = remove_columns)                              
    
    col_list = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount', 'logerror'] 
    # k value set to 3.0 to allow more outliers to be left 
    df = remove_outliers(df, 3.0, col_list)

    # add a column with information from yearbuilt column
    df['age'] = 2017 - df.yearbuilt
    df['taxrate'] = df['taxamount'] / df['taxvaluedollarcnt']
    df['dollars_per_sqft'] = df['taxvaluedollarcnt'] / df['calculatedfinishedsquarefeet']
    df = age_bin(df)
    df['abs_logerror'] = df['logerror'].abs()
    return df

######################################

def handle_missing_values(df, prop_required_column = .55, prop_required_row = .7):
    ''' 
    Takes in a datafram and is defaulted to have at least 55 percent of values for columns and 70 percent for rows
    '''
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df

######################################

def nulls_by_col(df):
    ''' 
    Takes in a dataframe and will output a dataframe of information for columns missing
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows':num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing

######################################

def nulls_by_row(df):
    ''' 
    Takes in a dataframe and will output a dataframe of information for rows missing
    '''
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing':num_cols_missing, 'pct_cols_missing':pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

######################################

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

######################################

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

######################################

def handle_nulls(train, validate, test):
    ''' 
    This function will take in your split data and handle the continuous columns with the mode
    and will handle the categorical columns with the median
    '''
    # continuous values filled with mode
    cols = [
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
    'taxamount',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'structuretaxvaluedollarcnt',
    'calculatedfinishedsquarefeet',
    'lotsizesquarefeet'
    ]

    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace = True)
        validate[col].fillna(median, inplace = True)
        test[col].fillna(median, inplace = True)

    return train, validate, test

######################################

def absolute_logerror(df):
    '''
    This function creates another column called abs_logerror from the logerror column
    '''
    df['abs_logerror'] = df['logerror'].abs()    
    return df

######################################

def age_bin(df):
    '''
    Function takes in a dataframe, uses the 'yearbuilt' column to create age bins
    pre 1970, 1970-2000, and post 2000 
    '''
    # set bin sizes
    year_bins = [df['yearbuilt'].min(), 1970, 2000,df['yearbuilt'].max()]
    
    # use cut to assign bins using yearbuilt column
    df['age_bin'] = pd.cut(df['yearbuilt'], year_bins)
    
    return df

######################################

def target_regplot(df, target, var_list, figsize = (8,5), hue = None):
    '''
    Takes in dataframe, target and varialbe list, and plots against target. 
    '''
    for var in var_list:
        plt.figure(figsize = (figsize))
        sns.regplot(data = df, x = var, y = target, 
                line_kws={'color': 'purple'})
        plt.tight_layout()

######################################

def show_pairplot(df):
    col_list = ['bathroomcnt','bedroomcnt', 'calculatedfinishedsquarefeet', 'dollars_per_sqft','latitude','longitude',
            'taxvaluedollarcnt','taxamount', 'age','taxrate','abs_logerror', 'logerror']

    return sns.pairplot(data = df[col_list], corner=True)

######################################

def county_plots(df):
    cols = [ 'age', 'latitude', 'longitude', 'dollars_per_sqft', 'calculatedfinishedsquarefeet', 'county']
    sns.pairplot(data = df[cols], hue = 'county', corner = True)

######################################

def get_median_baseline(train, validate, test):
    '''
    function takes in train validate test.
    Adds column with the baseline predictions based on the mean of train to each dataframe.
    returns train, validate, test
    '''
    train['baseline'] = train.logerror.median()
    validate['baseline'] = validate.logerror.median()
    test['baseline'] = validate.logerror.median()
    
    return train, validate, test

######################################

def scale_this(X_data, scalertype):
    '''
    X_data = dataframe with specified columns 
    scalertype = either StandardScaler() or MinMaxScaler()
    This function takes a dataframe (X_data), a scaler, and ouputs a new dataframe with those columns scaled. 
    And a scaler for inverse transformation
    '''
    scaler = scalertype.fit(X_data)

    X_scaled = pd.DataFrame(scaler.transform(X_data), columns = X_data.columns).set_index([X_data.index.values])
    
    return X_scaled, scaler

######################################

def get_X_train_y_train(X_cols, y_col, train, validate, test):
    '''
    columns should be scaled for the modeling
    X_cols = list of columns to be used as features
    y_col = name of the target column
    train = train dataframe
    validate = validate dataframe
    test = test dataframe
    returns X_train, y_train, X_validate, y_validate, X_test, and y_test
    '''  
    # X is the data frame of the features, y is a series of the target
    X_train, y_train = train[X_cols], train[y_col]
    X_validate, y_validate = validate[X_cols], validate[y_col]
    X_test, y_test = test[X_cols], test[y_col]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

######################################

def get_zillow_dummies(train, validate, test, cat_columns = ['age_bin', 'county', 'lat_long_age_cluster']):
    '''
    This function takes in train, validate, test and a list of categorical columns for dummies (cat_columns)
    default col_list is for zillow 
    '''
    # create dummies 
    train = pd.get_dummies(data = train, columns = cat_columns, drop_first=False)
    validate = pd.get_dummies(data = validate, columns = cat_columns, drop_first=False)
    test = pd.get_dummies(data = test, columns = cat_columns, drop_first=False)


    # drop columns I don't want (specified above)
    train = train.drop(columns=['age_bin_(1970.0, 2000.0]', 'county_Ventura', 'lat_long_age_cluster_0'])
    validate = validate.drop(columns=['age_bin_(1970.0, 2000.0]', 'county_Ventura', 'lat_long_age_cluster_0'])
    test = test.drop(columns=['age_bin_(1970.0, 2000.0]', 'county_Ventura', 'lat_long_age_cluster_0'])

    # rename age bins because they have a dumb name
    train = train.rename(columns={'age_bin_(1920.0, 1970.0]': 'built_before_1970', 
                                  'age_bin_(2000.0, 2016.0]': 'built_after_2000'})
    validate = validate.rename(columns={'age_bin_(1920.0, 1970.0]': 'built_before_1970', 
                                  'age_bin_(2000.0, 2016.0]': 'built_after_2000'})
    test = test.rename(columns={'age_bin_(1920.0, 1970.0]': 'built_before_1970', 
                                  'age_bin_(2000.0, 2016.0]': 'built_after_2000'})
    
    return train, validate, test


######################################

def plot_inertia(X_data, k_range_start = 1, k_range_end = 10):
    '''
    This function takes in a dataframe (must be scaled)
    Plots the change in inertia with 'x' markers
    Optional argument to adjust the range, default range(2,10)
    '''
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(10, 7))
        pd.Series({k: KMeans(k).fit(X_data).inertia_ for k in range(k_range_start, k_range_end)}).plot(marker='x')
        plt.xticks(range(k_range_start -1, k_range_end))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

######################################

def scatterplot_clusters(x ,y, cluster_col_name, df , kmeans, scaler, centroids):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""

    # set figsize
    plt.figure(figsize=(10, 6))
    
    # scatterplot the clusters 
    sns.scatterplot(x = x, y = y, data = df, hue = cluster_col_name, palette = 'cubehelix_r')
    
    # plot the centroids as Xs
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.60, s=500, c='black', marker = 'x')

######################################

def cluster_creator(X_data, k, col_name = None ):
    '''
    Function takes in scaled dataframe, k (number of clusters desired)
    Optional arguemenet col_name, If none is entered column returned is {k}_k_clusters
    Returns dataframe with column attached and dataframe with centroids (scaled) in it
    Returns: X_data, centroids_scaled, kmeans
    Use for exploring and when you need centroids
    '''
    
    # make thing
    kmeans = KMeans(n_clusters=k, random_state=123)

    # Fit Thing
    kmeans.fit(X_data)
    
    # create clusters
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = list(X_data))
    
    if col_name == None:
    #clusters on dataframe 
        X_data['clusters'] = kmeans.predict(X_data)
        X_data.clusters = X_data.clusters.astype('category')
    else:
        X_data[col_name] = kmeans.predict(X_data)
        X_data[col_name] = X_data[col_name].astype('category')
    
    return X_data, centroids_scaled, kmeans

######################################

def scale_this2(train, validate, test, cont_columns, scaler): 
    '''
    This function takes in train, validate, and test, columns desired to be scaled (as a list), 
    a scaler (i.e. MinMaxScaler(), with parameters needed),
    cont_columns: list of columns for scaling
    Replaces columns with scaled columns 
    Outputs scaler for doing inverse transforms.
    '''
    # create scaler (minmax scaler)
    minmax_scaler = scaler
    
    # loop through columns in columns_list
    for col in cont_columns:
        # fit and transform to train, add new columns on train df
        train[f'{col}'] = minmax_scaler.fit_transform(train[[col]]) 
        
        # transform columns from validate / test (only fit on train)
        validate[f'{col}']= minmax_scaler.transform(validate[[col]])
        test[f'{col}']= minmax_scaler.transform(test[[col]])

    # returns scaler and a list of columns to be used for X_train, X_validate, X_test
    return train, validate, test, scaler 

######################################

def create_clusters(train, validate, test, k, cols_for_cluster, col_name = None ):
    '''
    This function takes in scaled train, validate, and test
    k (number of clusters desired) and  cols_for_cluster that has been created specifically for clustering
    Optional argumenet col_name, none is defaulted so column returned is 'clusters'
    Returned dataframes with column attached  and kmeans
    Returns: train, validate, test, kmeans
    Use for making some magic
    ''' 
    # make thing
    kmeans = KMeans(n_clusters=k, random_state=123)

    #Fit Thing
    kmeans.fit(train[cols_for_cluster])
    
    if col_name == None:
        # add cluster predictions on dataframe generic
        train['clusters'] = kmeans.predict(train[cols_for_cluster])
        train.clusters = train.clusters.astype('category')
        validate['clusters'] = kmeans.predict(validate[cols_for_cluster])
        validate.clusters = validate.clusters.astype('category')
        test['clusters'] = kmeans.predict(test[cols_for_cluster])
        test.clusters = test.clusters.astype('category')
    else:
        # add cluster predictions on dataframe specific name
        train[col_name] = kmeans.predict(train[cols_for_cluster])
        train[col_name] = train[col_name].astype('category')
        validate[col_name] = kmeans.predict(validate[cols_for_cluster])
        validate[col_name] = validate[col_name].astype('category')
        test[col_name] = kmeans.predict(test[cols_for_cluster])
        test[col_name] = test[col_name].astype('category')
        
    return train, validate, test, kmeans

######################################
# Create function to do seperate dataframes based on location
def location_location(df):
    '''
    '''
    # old
    df_LA = df[df['county_Los_Angeles'] == 1]
    
    # new
    df_not_LA = df[df['county_Los_Angeles'] == 0]
    
    return df_LA, df_not_LA

###################################### model prep ######################################

def some_magic():
        '''
        This function was created for when i am ready to move into the modeling phase.
        The function will start fresh with the original dataset and made changes that i have found to
        be helpful from my exploration phase.
        This function will output train, validate, test and create clusters.
        unneccessry columns will be dropped and dummy columns for categorical columns.
        scaler , train, validate, and test will be returned
        '''

        # Define unneaded columns for dropping later

        dropping_cols = ['lotsizesquarefeet', 'regionidcity', 'regionidzip', 'unitcnt', 'regionidcounty', 'yearbuilt',
        'censustractandblock', 'transactiondate', 'heatingorsystemdesc', 'propertylandusedesc' ]

        # get the zillow data from wrangle zillow
        df = get_zillow_data()

        df = clean_zillow(df)

        df = absolute_logerror(df)

        train, validate, test = split_data(df)

        train, validate, test = handle_nulls(train, validate, test)

        cont_columns = ['bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet', 'dollars_per_sqft', 
            'latitude', 'longitude', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 
            'taxamount','age', 'taxrate']

        cat_columns = ['age_bin', 'county', 'lat_long_age_cluster']

        cols_for_cluster = ['latitude', 'longitude', 'age']

        train, validate, test, scaler = scale_this2(train, validate, test, cont_columns, MinMaxScaler())

        train, validate, test, kmeans = create_clusters(train, validate, test, 4, cols_for_cluster, col_name = 'lat_long_age_cluster')

        train, validate, test = get_zillow_dummies(train, validate, test)

        return train, validate, test, scaler




        