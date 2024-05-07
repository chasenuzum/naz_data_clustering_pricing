# # Exploratory Data Analysis (EDA)
# 
# ## Overview
# Exploratory Data Analysis (EDA) is to utilize pandas, numpy, plotting libraries to understand dataset better and let data guide in subsequent modeling steps.
# 
# ## Data Summary
# - **Dataset Description**: The dataset includes information about product sales, store characteristics, and other relevant variables.
# - **Features**: 
#   - `product_key`: ID for different products (SKUs)
#   - `unit_sales`: Number of products sold
#   - `dollar_sales`: Revenue of the product
#   - `volume_sales`: Product sales volume (ounces)
#   - `brewer`: Name of the brewer (AB vs OTHER)
#   - `wholesaler_id_value`: Wholesaler ID
#   - `retailer_store_number`: Retailer ID
#   - `package_value`: Package info of the product
#   - `brand_value`: Product brand
#   - `year_week`: Time variable
# 
# ## Data Exploration Steps
# 1. **Load the Dataset**: 
#    - Import the dataset into a Pandas DataFrame.
#    
# 2. **Data Cleaning**:
#    - Check for missing values and handle them appropriately (imputation, removal, etc.).
#    - Check for duplicates and remove them if necessary.
#    
# 3. **Descriptive Statistics**:
#    - Compute summary statistics (mean, median, min, max, etc.) for numerical features to understand their distribution.
#    - Examine unique values and frequency counts for categorical features.
#    
# 4. **Visualization**:
#    - Create visualizations such as histograms, box plots, and scatter plots to explore the distributions and relationships between variables.
#    - Visualize trends over time using line plots or time series plots for relevant features (e.g., dollar_sales over time).
#    
# 5. **Identify Outliers**:
#    - Detect outliers using statistical methods or visualization techniques.
#    - Decide whether to remove or treat outliers based on their impact on the analysis.
#    
# 6. **Correlation Analysis**:
#    - Compute correlation coefficients between numerical features to identify relationships.
#    - Visualize correlations using heatmaps or pair plots.
#    
# 7. **Segmentation Analysis**:
#    - Explore potential segmentation variables (e.g., store characteristics, product attributes) to understand variations in the data.
#    - Visualize the distribution of key variables across segments.
# 
# ## Summary
# Exploratory Data Analysis provides valuable insights into the dataset, including its structure, distributions, and relationships. By understanding these aspects of the data, we can better inform subsequent analyses and decision-making processes related to price optimization and product recommendation for Anheuser-Busch products at retail stores.
# 


# load in pacakges based on requirements provided
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils import remove_outliers

# set to base working directory of project based on file location
base_dir = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
os.chdir(base_dir)

# remove scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset
df = pd.read_csv('data/beer_data.csv')

# remove columns that entirely NaN
df = df.dropna(axis=1, how='all')
df.head(10)


# check NaN counts in each column, mostly all here; will need to impute
df.isna().sum()

df.fillna('OTHER', inplace=True) # assume missing brewer is 'Other' also...


# check df types iteratively
for i in df.columns:
    print(i, df[i].dtype)


# create date col to datetime (year_week), can use later for aggregation or time series analysis, format is year and week number like 201623
df['date'] = pd.to_datetime(df['year_week'].astype('str') + '1', format='%Y%W%w') # assume first day of week for sales collection
df['date'].head() # check that date column was created


# run descriptive statistics on numerical columns
df.describe()

# see date boundaries, spread of sales and volume (units sold) variance, std shows skewness will need to check


# find number of retailers
print(df['wholesaler_id_value'].nunique()) # 10 retailers
print(df['retailer_store_number'].nunique()) # 10 retailers



# volume sales being zero is a bit odd, but could be due to missing data or other reasons, will need to check or fill with proxy
df.loc[df['volume_sales'] <= 0.0, 'volume_sales'] = .001 # fill with proxy value


NUM_COLS = ['dollar_sales', 'unit_sales', 'volume_sales']


# check for outliers in numerical columns of note
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
for i in enumerate(NUM_COLS):
    ax[i[0]].hist(df[i[1]], bins=50)
    ax[i[0]].set_title(i[1] + ' Histogram')


# check for outliers in numerical columns of note
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
for i in enumerate(NUM_COLS):
    ax[i[0]].boxplot(df[i[1]])
    ax[i[0]].set_title(i[1] + ' Boxplot')


# skew present so take log of numerical columns; some values close to zero so add 1 to avoid log(0)
df[NUM_COLS] = np.log1p(df[NUM_COLS])


# check for outliers in numerical columns of note
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
for i in enumerate(NUM_COLS):
    ax[i[0]].hist(df[i[1]], bins=50)
    ax[i[0]].set_title(i[1] + ' Histogram')

# better looking distributions, now check for outliers again with IQR


# based on charts check percentage outliers in each column
def iqr_count(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].shape[0]

for i in NUM_COLS:
    print('Outliers percent in', i, ': %', round(iqr_count(df, i) / len(df), 3) * 100) # large amount of outliers in dollar_sales and unit_sales as percent of total
# outliers also now closer to zero, good for modeling


# check correlation between numerical columns (multicollinearity could exist)
correl = df.loc[:,NUM_COLS].corr().round(3)
correl.to_csv('data/correlation.csv')
# > .5 across the board
correl


# qual check counts of categorical columns of importance
CAT_COLS = ['city', 'state_code', 'Package_Value', 'BRAND_VALUE', 'Brewer']


# check for unique values in each column
for i in CAT_COLS:
    print(i, df[i].nunique())


# print value count as ratio of total
for i in CAT_COLS:
    vc = df[i].value_counts(normalize=True).round(3)
    vc.to_csv('data/' + i + '_value_counts.csv')
    print(i,vc)


# plotly express bar chart for categorical columns
for i in CAT_COLS:
    fig = px.bar(df[i].value_counts(normalize=True), title=i)
    fig.show()


# check seasonality in units sold and dollar sales
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# check seasonality and trend in units sold and dollar sales
fig = px.line(df.groupby('month').agg({'dollar_sales': 'sum'}).reset_index(), x='month', y='dollar_sales', title='Dollar Sales by Month')
fig.show()

fig = px.line(df.groupby('year').agg({'unit_sales': 'sum'}).reset_index(), x='year', y='unit_sales', title='Unit Sales by Year')
fig.show()


# create some simple features for clustering, could be useful for segmentation
df['dollar_per_unit'] = df['dollar_sales'] / df['unit_sales']
df['dollar_per_oz'] = df['dollar_sales'] / df['volume_sales']
df['unit_size_oz'] = df['volume_sales'] / df['unit_sales']

# calculate different features for retailers; which is the pricing mechanism
df['sku_sales'] = df.groupby('Product_Key')['unit_sales'].transform('sum')  # could be useful for segmentation
df['retailer_sales'] = df.groupby('retailer_store_number')['unit_sales'].transform('sum')  # could be useful for segmentation


# calculate price elasticity of demand for each product; more feature generation
elasticity_df = df.groupby(['Product_Key', 'date']).agg({'unit_sales': 'sum', 'dollar_sales': 'sum'}).reset_index()
# create key for later
elasticity_L = list()
# Calculate percentage change in price elasticity of deamand for each product, need a key for later 
for product in elasticity_df['Product_Key'].unique():
    sub_df = elasticity_df.loc[elasticity_df['Product_Key'] == product]
    sub_df['unit_pct_change'] = sub_df.loc[:,'unit_sales'].pct_change()
    sub_df['sales_pct_change'] = sub_df.loc[:,'dollar_sales'].pct_change()
    sub_df['elasticity'] = sub_df['unit_pct_change'] / sub_df['sales_pct_change']
    sub_df = sub_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()
    elasticity_L.append(sub_df['elasticity'].median()) # use median as it is more robust to outliers

elasticity_key = pd.DataFrame(elasticity_L, columns=['elasticity'],
                              index=elasticity_df['Product_Key'].unique())
elasticity_key.index.rename('Product_Key')


elasticity_key.sort_values('elasticity', ascending=False).head(10) # appears to be error in some calcs, but some products have high elasticity


elasticity_key.sort_values('elasticity', ascending=True).head(10) # appears to be error in some calcs, but some products have high elasticity


# clean outliers, probably due to data quality issues
elasticity_key = elasticity_key.loc[(elasticity_key['elasticity'] < 2) & (elasticity_key['elasticity'] > 0.0) ] # remove outliers; lack observed elasticity causing issues with some


price_fig = elasticity_key.sort_values('elasticity', ascending=False)
price_fig.index = price_fig.index.astype('str')
fig = px.bar(price_fig[:10], y='elasticity', title='Top 10 Most Elastic (Price Sensitive) Product')
fig.show()
fig = px.bar(price_fig[-10:], y='elasticity', title='Bottom 10 Most Inelastic Product')
fig.show()


# merge elasticity key back to main df
df = df.merge(elasticity_key, left_on='Product_Key', right_index=True, how='left')


# create date based features to solve products not sold in last 6 months; and date based features for clustering
df['last_sale'] = df.groupby('Product_Key')['date'].transform('max')
# months since last sale
df['months_since_last_sale'] = (pd.to_datetime('today') - df['last_sale']).dt.days / 30

# create product age, in years
df['product_age'] = (df['date'] - df.groupby('Product_Key')['date'].transform('min')).dt.days / 365


# add our new numerical columns to the list, features for later clustering
for i in ['dollar_per_unit', 'dollar_per_oz', 'unit_size_oz', 'sku_sales', 'retailer_sales', 'product_age', 'elasticity']:
    NUM_COLS.append(i)


# create some n quantile features for clustering, use strings for names to avoid confusion
for i in NUM_COLS:
    try:
        df[i + '_quantile'] = pd.qcut(df[i], 5, labels=['lowest', 'low', 'mid', 'high', 'highest'])
    except ValueError:
        print(i, 'cannot be quantiled to 5 bins')
# tried by removed quantiles for now


df.to_csv('data/processed_data.csv') # processed for product analysis and clustering


df.head()


# create product composition features for each retailer
top_brands = df.groupby(['retailer_store_number', 'BRAND_VALUE']).agg({'unit_sales': 'sum'}).reset_index()
# get top brand for each retailer
top_brands = top_brands.loc[top_brands.groupby('retailer_store_number')['unit_sales'].idxmax()]
top_brands = top_brands.rename(columns={'BRAND_VALUE': 'top_brand', 'unit_sales': 'top_brand_sales'})
top_brands.to_csv('data/top_brands.csv')
# calc & of total sales
print(top_brands)


# join top brands back to main df
df_new = df.merge(top_brands, on='retailer_store_number', how='left')
print(df_new.head())


df_new.to_csv('data/processed_data.csv') # processed for product analysis and clustering


# create seller dataset with features for clustering
seller_df = df_new.groupby(['retailer_store_number', 'city', 'top_brand', 'date']).agg({'dollar_sales': 'sum', 
                                                                                    'unit_sales': 'sum', 
                                                                                    'volume_sales': 'sum',
                                                                                    'dollar_per_unit': 'mean', 
                                                                                    'dollar_per_oz': 'mean', 
                                                                                    'unit_size_oz': 'mean',
                                                                                    'sku_sales': 'sum', 
                                                                                    'retailer_sales': 'mean', 
                                                                                    'months_since_last_sale': 'mean',
                                                                                    'product_age': 'mean',
                                                                                     'elasticity' : 'mean', 
                                                                                     'top_brand_sales' : 'mean'}).reset_index()


print(seller_df)


seller_df.to_csv('data/seller_data.csv') # processed for seller analysis and clustering


# find if multiple exist for each retailer
seller_count = seller_df.groupby('retailer_store_number')['city'].nunique()


# 23 physical stores, 10 retailers, so some have multiple locations
seller_count.sum()