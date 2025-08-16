#!/usr/bin/env python3
"""
COVID-19 Data Visualization Project
Main script to run the complete analysis
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data_loader import COVIDDataLoader
from src.data_processor import COVIDDataProcessor
from src.visualizations import COVIDVisualizations

def main():
    print("🦠 COVID-19 Data Visualization Project")
    print("=" * 50)
    
    # Initialize components
    loader = COVIDDataLoader()
    visualizer = COVIDVisualizations()
    
    # Create output directory
    os.makedirs('output/plots', exist_ok=True)
    
    # Step 1: Download data
    print("\n📥 Step 1: Downloading COVID-19 data...")
    loader.download_johns_hopkins_data()
    loader.download_owid_data()
    
    # Step 2: Load data
    print("\n📊 Step 2: Loading datasets...")
    datasets = loader.load_data()
    print(f"Loaded {len(datasets)} datasets")
    
    if len(datasets) == 0:
        print("❌ No datasets loaded. Exiting...")
        return
    
    # Step 3: Process data
    print("\n🔄 Step 3: Processing data...")
    processor = COVIDDataProcessor(datasets)
    processed_data = processor.process_johns_hopkins_data()
    owid_data = processor.process_owid_data()
    
    print(f"Processed datasets: {list(processed_data.keys())}")
    
    # Calculate additional metrics for available datasets
    for data_type in processed_data:
        if len(processed_data[data_type]) > 0:
            processed_data[data_type] = processor.calculate_daily_changes(
                processed_data[data_type], data_type
            )
            processed_data[data_type] = processor.calculate_moving_average(
                processed_data[data_type], data_type
            )
    
    # Step 4: Create visualizations
    print("\n📈 Step 4: Creating visualizations...")
    
    if len(processed_data) > 0:
        # Global trends
        print("Creating global trends visualization...")
        global_data = processor.create_global_summary(processed_data)
        visualizer.plot_global_trends(
            global_data, 
            save_path='output/plots/global_trends.png'
        )
        
        # Top countries analysis for available metrics
        for metric in processed_data.keys():
            if len(processed_data[metric]) > 0:
                print(f"Creating top countries analysis for {metric}...")
                visualizer.plot_top_countries(
                    processed_data, 
                    metric=metric,
                    save_path=f'output/plots/top_countries_{metric}.png'
                )
        
        # Comprehensive dashboard
        print("Creating comprehensive dashboard...")
        visualizer.create_summary_dashboard(
            processed_data,
            owid_data,
            save_path='output/plots/covid_dashboard.png'
        )
    
    # OWID-specific visualizations
    if owid_data is not None and len(owid_data) > 0:
        print("Creating correlation heatmap...")
        visualizer.plot_correlation_heatmap(
            owid_data,
            save_path='output/plots/correlation_heatmap.png'
        )
        
        print("Creating vaccination progress chart...")
        visualizer.plot_vaccination_progress(
            owid_data,
            save_path='output/plots/vaccination_progress.png'
        )
    
    print("\n✅ Analysis complete!")
    output_files = [f for f in os.listdir('output/plots') if f.endswith('.png')]
    print(f"📁 All visualizations saved to: output/plots/")
    print(f"🎯 Generated visualizations:")
    for file in output_files:
        print(f"   • {file}")

if __name__ == "__main__":
    main()