import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class COVIDDataProcessor:
    def __init__(self, datasets):
        self.datasets = datasets
        
    def process_johns_hopkins_data(self):
        """Process Johns Hopkins time series data"""
        processed_data = {}
        
        for data_type in ['confirmed', 'deaths', 'recovered']:
            if data_type not in self.datasets:
                continue
                
            df = self.datasets[data_type].copy()
            
            # Melt the dataframe to long format
            id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
            df_melted = df.melt(id_vars=id_vars, var_name='Date', value_name=data_type)
            
            # Convert date column
            df_melted['Date'] = pd.to_datetime(df_melted['Date'])
            
            # Group by country and date
            df_country = df_melted.groupby(['Country/Region', 'Date'])[data_type].sum().reset_index()
            
            processed_data[data_type] = df_country
        
        return processed_data
    
    def create_global_summary(self, processed_data):
        """Create global summary statistics"""
        global_data = []
        
        for data_type, df in processed_data.items():
            global_summary = df.groupby('Date')[data_type].sum().reset_index()
            global_summary['Type'] = data_type
            global_data.append(global_summary)
        
        return pd.concat(global_data, ignore_index=True)
    
    def calculate_daily_changes(self, df, value_col):
        """Calculate daily new cases/deaths"""
        df = df.sort_values(['Country/Region', 'Date'])
        df['Daily_New'] = df.groupby('Country/Region')[value_col].diff().fillna(0)
        df['Daily_New'] = df['Daily_New'].clip(lower=0)  # Remove negative values
        return df
    
    def calculate_moving_average(self, df, value_col, window=7):
        """Calculate moving average for smoother trends"""
        df = df.sort_values(['Country/Region', 'Date'])
        df[f'{value_col}_MA{window}'] = df.groupby('Country/Region')[value_col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        return df
    
    def get_top_countries(self, df, value_col, n=10, date=None):
        """Get top N countries by specified metric"""
        if date is None:
            # Use latest date
            latest_data = df.loc[df.groupby('Country/Region')['Date'].idxmax()]
        else:
            latest_data = df[df['Date'] == date]
        
        return latest_data.nlargest(n, value_col)
    
    def process_owid_data(self):
        """Process Our World in Data comprehensive dataset"""
        if 'owid' not in self.datasets:
            return None
            
        df = self.datasets['owid'].copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Select relevant columns
        relevant_cols = [
            'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 
            'new_deaths', 'total_cases_per_million', 'new_cases_per_million',
            'total_deaths_per_million', 'new_deaths_per_million', 'population',
            'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated'
        ]
        
        available_cols = [col for col in relevant_cols if col in df.columns]
        df_processed = df[available_cols].copy()
        
        return df_processed