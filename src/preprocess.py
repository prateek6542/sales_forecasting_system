import pandas as pd

def load_data(file_path):
    
    #Load data from a CSV file.
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    
    #clean and prepare dataset
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    #Sort Values
    
    df = df.sort_values(by='date')
    return df