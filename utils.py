import requests
import json
import pandas as pd

def getfreddata(fredid, apikey):
    """get data from FRED via API with item code, fredid, and apikey"""
    url = r'https://api.stlouisfed.org/fred/series/observations' + \
    '?series_id='  + fredid + \
    '&api_key=' + apikey + \
    '&file_type=json'
    x = requests.get(url)
    data = json.loads(x.text)
    df = pd.DataFrame(data['observations'])
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

# Handle outliers using IQR method
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
