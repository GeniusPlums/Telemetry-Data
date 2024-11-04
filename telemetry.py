import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
data = pd.DataFrame({
    'car_id': [f'C{i}' for i in range(1, 101)],
    'average_speed_kmh': [131,96,80,62,94,87,62,82,75,84,90,92,102,68,96,101,72,86,65,72,80,73,99,85,133,53,88,62,79,87,99,66,75,67,87,58,98,76,45,77,69,80,101,73,76,93,86,71,123,96,74,76,63,76,116,90,88,79,66,57,136,79,62,83,75,65,86,81,68,89,65,69,127,80,65,97,63,93,65,86,77,73,79,113,120,77,64,92,59,69,87,97,95,82,68,90,130,97,71,105],
    'hard_brakes': [9,4,9,4,2,4,0,9,7,9,8,3,7,2,8,0,2,8,3,9,1,5,8,1,6,4,8,1,4,5,1,2,5,0,5,5,0,6,1,0,9,5,5,2,8,5,3,1,2,1,8,6,9,4,0,2,8,8,1,8,1,0,0,6,4,7,9,1,0,4,4,3,4,5,0,2,7,0,8,4,7,1,0,8,0,0,4,7,0,7,3,0,9,5,0,0,5,5,3,9],
    'driving_time_min': [105,36,89,39,68,57,53,76,99,95,36,93,67,108,95,38,68,100,38,37,82,35,92,59,76,78,53,91,92,116,76,41,43,86,76,42,80,61,36,53,88,78,52,65,59,101,95,68,58,114,55,31,96,111,71,45,45,39,60,62,39,36,44,48,97,48,86,58,99,110,96,91,105,100,47,55,54,108,71,38,101,49,73,68,66,34,41,34,102,36,83,82,96,111,84,78,82,36,104,67],
    'night_driving_ratio': [np.nan,0.92,0.48,0.71,0.79,0.35,0.54,0.49,np.nan,0.69,0.45,0.56,0.61,0.68,0.16,0.62,np.nan,0.65,0.78,0.57,0.37,0.3,0.69,0.41,np.nan,0.72,0.81,0.79,0.88,0.11,0.23,0.71,np.nan,0.83,0.59,0.97,0.98,0.24,0.2,0.8,np.nan,0.29,0.72,0.47,0.16,0.76,0.31,0.12,np.nan,0.93,0.53,0.06,0.38,0.86,0.3,0.77,np.nan,0.93,0.15,0.6,0.27,0.62,0.09,0.16,np.nan,0.12,0.51,0.02,0.37,0.97,0.65,0.67,np.nan,0.29,0.59,0.39,0.26,0.65,0.08,0,np.nan,0.47,0.92,0.74,0.49,0.39,0.52,0.99,np.nan,0.8,0.75,0.8,0.12,0.74,0.99,0.8,np.nan,0.7,0.28,0.17],
    'maintenance_flag': [0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0]
})

# Function to identify outliers using IQR method
def remove_outliers(df, columns):
    df_clean = df.copy()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]['car_id'].tolist()
        if outliers:
            print(f"\nOutliers in {column}:")
            print(f"Cars: {', '.join(outliers)}")
            print(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    return df_clean

# Remove outliers from numerical columns
numerical_columns = ['average_speed_kmh', 'hard_brakes', 'driving_time_min']
data_clean = remove_outliers(data, numerical_columns)

# Hypothesis Testing
def perform_hypothesis_tests(df):
    results = []
    
    # Test 1: Is there a relationship between maintenance flags and average speed?
    t_stat, p_val = stats.ttest_ind(
        df[df['maintenance_flag'] == 1]['average_speed_kmh'],
        df[df['maintenance_flag'] == 0]['average_speed_kmh']
    )
    results.append(("Average Speed vs Maintenance", p_val))
    
    # Test 2: Is there a relationship between maintenance flags and hard brakes?
    t_stat, p_val = stats.ttest_ind(
        df[df['maintenance_flag'] == 1]['hard_brakes'],
        df[df['maintenance_flag'] == 0]['hard_brakes']
    )
    results.append(("Hard Brakes vs Maintenance", p_val))
    
    # Test 3: Is there a correlation between night driving and maintenance?
    t_stat, p_val = stats.ttest_ind(
        df[df['maintenance_flag'] == 1]['night_driving_ratio'].dropna(),
        df[df['maintenance_flag'] == 0]['night_driving_ratio'].dropna()
    )
    results.append(("Night Driving vs Maintenance", p_val))
    
    print("\nHypothesis Test Results (α = 0.05):")
    for test, p_value in results:
        print(f"{test}: p-value = {p_value:.4f} ({p_value < 0.05})")

# Perform hypothesis tests
perform_hypothesis_tests(data_clean)

# Calculate key statistics
def calculate_statistics(df):
    print("\nKey Statistics:")
    print(f"Total vehicles analyzed: {len(df)}")
    print(f"Vehicles requiring maintenance: {df['maintenance_flag'].sum()} ({(df['maintenance_flag'].sum()/len(df)*100):.1f}%)")
    print(f"Average speed: {df['average_speed_kmh'].mean():.1f} km/h")
    print(f"Average hard brakes per trip: {df['hard_brakes'].mean():.1f}")
    print(f"Average driving time: {df['driving_time_min'].mean():.1f} minutes")
    print(f"Average night driving ratio: {df['night_driving_ratio'].mean():.2f}")

calculate_statistics(data_clean)
