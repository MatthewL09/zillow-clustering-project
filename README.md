PROJECT DETAILS FOR ZILLOW DATASET

1. Overview Project Goals

    - Continue working with the previous project and incorporate clustering methodologies
    - Construct a machine learning regression model.
    - Find key drivers of the error of the zestimate.

2. Project Description
    - Include clustering methodologies to further your findings

3. Initial Questions/Hypothesis

    - Does location affect tax value?
    - Does square footage affect tax value?
    - Does lot size affect tax value?
    - Does bedroom count affect tax value?
    - Does bathroom count affect tax value?

4. Data Dictionary 
   |Column | Description | Dtype|
    |--------- | --------- | ----------- |
    bedroom | the number of bedrooms | int64 |
    bathroom | the number of bathrooms | int64 |
    square_ft | square footage of property | int64 |
    lot_size | square footage of lot | int 64 |
    tax_value | property tax value dollar amount | int 64 |
    year_built | year the property was built | int64 |
    fips | geo code of property | int64 |
    county | county the property is in | object |
    age | the difference between year_built and 2017 | int 64
    los_angeles | county name of geo code  | uint8 |
    orange | county name of geo code | uint8 |
    ventura | county name of geo code | uint8 |
    logerror | error in zestimate | int64

5. Project Planning

    - Recreate the project by following these steps
    
    Planning 
    - Define goals
    - Ask questions / formulate hypotheses
    - Clustering included to assist Supervised Learning
    - Determine audience and deliver format

    Acquisition
    - Create a function that establishes connection to the zillow_db
    - Create a function that holds your SQL query and reads the results
    - Create a function for caching data and stores as .csv for ease
    - Create and save in wrangle_zillow.py so functions can be imports
    - Test functions

    Preparation
    - Create a function thatpreps the acquired data
    - This function will:
        - remove duplicates
        - handle missing values
        - convert data types
        - handle outliers
        - encode categorical columns
        - renames columns
        - created a columns for house 'age'
        - scale data for exploration
    - Create a function that splits the data into 3 sets. train, validate, test
        - Split 20% (test data), 24% (validate data), and 56%(test data)
    - Create functions and save in wrangle_zillow.py to be easily imported
    - Test functions

    Exploration 
    - guide exploration with initial questions
    - Create visualizations of data
    - Statistical analysis to confirm or deny hypothesis
    - Save work with notations in zillow_prep.ipynb
    - Document answers to questions as Takeaways

    Model
    - The baseline and 4 different models were created

    Delivery
    - Report is saved in Jupyter Notebook
    - Present via Zoom
    - The audience is the data science team at Zillow

6. Recreation of Project:
    - You will need an env file with database credentials saved to your project directory
        - database credentials (username, password, hostname)
    - Create a gitignore with env file inside to prevent sharing of credentials
    - Push .py and .ipynb files to project directory
    - Create a final notebook to your project directory
    - Review this README.md
    - Libraries used are pandas, numpy, matplotlib, seaborn, Scipy, sklearn
    - Run zillow_final.ipynb

7. Key Findings and takeaways
    - 
    - Identified drivers of tax value as:
        - 
        - 
        - 
        - 
    - Recommendation: