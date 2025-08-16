import pandas as pd
import requests
import os
from datetime import datetime

class COVIDDataLoader:
    def __init__(self):
        self.base_urls = {
            'johns_hopkins': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/',
            'owid': 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/'
        }
        
    def download_johns_hopkins_data(self, data_dir='data/raw/'):
        """Download COVID-19 data from Johns Hopkins University"""
        files = {
            'confirmed': 'time_series_covid19_confirmed_global.csv',
            'deaths': 'time_series_covid19_deaths_global.csv',
            'recovered': 'time_series_covid19_recovered_global.csv'
        }
        
        os.makedirs(data_dir, exist_ok=True)
        
        for data_type, filename in files.items():
            url = self.base_urls['johns_hopkins'] + filename
            try:
                print(f"Downloading {data_type} data...")
                response = requests.get(url)
                response.raise_for_status()
                
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"✓ Downloaded {filename}")
                
            except requests.RequestException as e:
                print(f"✗ Error downloading {filename}: {e}")

                continue
    
    def download_owid_data(self, data_dir='data/raw/'):
        """Download comprehensive COVID-19 data from Our World in Data"""
        url = self.base_urls['owid'] + 'owid-covid-data.csv'
        filepath = os.path.join(data_dir, 'owid-covid-data.csv')
        
        try:
            print("Downloading OWID comprehensive data...")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("✓ Downloaded OWID data")
            
        except requests.RequestException as e:
            print(f"✗ Error downloading OWID data: {e}")
    
    def load_data(self, data_dir='data/raw/'):
        """Load all COVID-19 datasets"""
        datasets = {}
        
        # Load Johns Hopkins data
        jh_files = {
            'confirmed': 'time_series_covid19_confirmed_global.csv',
            'deaths': 'time_series_covid19_deaths_global.csv',
            'recovered': 'time_series_covid19_recovered_global.csv'
        }
        
        for data_type, filename in jh_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    datasets[data_type] = pd.read_csv(filepath)
                    print(f"✓ Loaded {data_type} data: {len(datasets[data_type])} rows")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
            else:
                print(f"⚠️  File not found: {filename}")
        
        # Load OWID data
        owid_filepath = os.path.join(data_dir, 'owid-covid-data.csv')
        if os.path.exists(owid_filepath):
            try:
                datasets['owid'] = pd.read_csv(owid_filepath)
                print(f"✓ Loaded OWID data: {len(datasets['owid'])} rows")
            except Exception as e:
                print(f"✗ Error loading OWID data: {e}")
        else:
            print("⚠️  OWID data file not found")
        
        return datasets