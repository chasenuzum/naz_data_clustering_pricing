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

# %%
# load in pacakges based on requirements provided
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# set to base working directory of project based on file location
base_dir = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
os.chdir(base_dir)
from utils import remove_outliers

# remove scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset
df = pd.read_parquet('data/beer_data.parquet')

# Remove columns that are entirely NaN
df = df.dropna(axis=1, how='all')

# Fill missing values with 'OTHER' for 'brewer' column
df['Brewer'].fillna('OTHER', inplace=True)

# Convert 'year_week' to datetime format
df['date'] = pd.to_datetime(df['year_week'].astype(str) + '1', format='%Y%W%w')

# Log transform numerical columns to handle skewness
num_cols = ['dollar_sales', 'unit_sales', 'volume_sales']
df[num_cols] = np.log1p(df[num_cols])

# Remove outliers from numerical columns
for col in num_cols:
    df = remove_outliers(df, col)

# Calculate correlation between numerical columns
correlation = df[num_cols].corr().round(3)

# Explore unique values and frequency counts for categorical columns
cat_cols = ['city', 'state_code', 'Package_Value', 'BRAND_VALUE', 'Brewer']
for col in cat_cols:
    print(df[col].value_counts(normalize=True).round(3))

# Visualize trends over time
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
plt.plot(df.groupby('month')['dollar_sales'].sum(), label='Dollar Sales by Month')
plt.legend()
plt.show()
plt.plot(df.groupby('year')['unit_sales'].sum(), label='Unit Sales by Year')
plt.legend()
plt.show()

# Create additional features for clustering
df['dollar_per_unit'] = df['dollar_sales'] / df['unit_sales']
df['dollar_per_oz'] = df['dollar_sales'] / df['volume_sales']
df['unit_size_oz'] = df['volume_sales'] / df['unit_sales']
df['sku_sales'] = df.groupby('Product_Key')['unit_sales'].transform('sum')
df['retailer_sales'] = df.groupby('retailer_store_number')['unit_sales'].transform('sum')

# Calculate price elasticity of demand
elasticity_df = df.groupby(['Product_Key', 'date']).agg({'unit_sales': 'sum', 'dollar_sales': 'sum'}).reset_index()
elasticity_df['elasticity'] = elasticity_df['unit_sales'].pct_change() / elasticity_df['dollar_sales'].pct_change()
elasticity_df['elasticity'] = elasticity_df['elasticity'].ffill()

# Merge elasticity back to main df
df = df.merge(elasticity_df[['Product_Key', 'elasticity']], on='Product_Key', how='left')

# Create date-based features
df['last_sale'] = df.groupby('Product_Key')['date'].transform('max')
df['months_since_last_sale'] = (pd.to_datetime('today') - df['last_sale']).dt.days / 30
df['product_age'] = (df['date'] - df.groupby('Product_Key')['date'].transform('min')).dt.days / 365

# Export processed data to CSV
df.to_parquet('data/processed_data.parquet')

# Create product composition features for each retailer
top_brands = df.groupby(['retailer_store_number', 'BRAND_VALUE']).agg({'unit_sales': 'sum'}).reset_index()
top_brands = top_brands.loc[top_brands.groupby('retailer_store_number')['unit_sales'].idxmax()]
top_brands.to_csv('data/top_brands.csv')

# Create seller dataset with features for clustering
seller_df = df.groupby(['retailer_store_number', 'city', 'top_brand', 'date']).agg({
    'dollar_sales': 'sum', 'unit_sales': 'sum', 'volume_sales': 'sum',
    'dollar_per_unit': 'mean', 'dollar_per_oz': 'mean', 'unit_size_oz': 'mean',
    'sku_sales': 'sum', 'retailer_sales': 'mean', 'months_since_last_sale': 'mean',
    'product_age': 'mean', 'elasticity': 'mean', 'top_brand_sales': 'mean'
}).reset_index()
seller_df.to_csv('data/seller_data.csv')

# Check if multiple locations exist for each retailer
seller_count = seller_df.groupby('retailer_store_number')['city'].nunique()