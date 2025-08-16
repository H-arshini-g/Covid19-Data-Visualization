import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class COVIDVisualizations:
    def __init__(self):
        # Set style - handle different seaborn versions
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
        
    def plot_global_trends(self, global_data, save_path=None):
        """Plot global COVID-19 trends over time"""
        # Determine available data types
        available_types = global_data['Type'].unique()
        
        # Create subplot layout based on available data
        if len(available_types) == 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        
        fig.suptitle('Global COVID-19 Trends Over Time', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Plot cumulative cases
        if 'confirmed' in available_types:
            confirmed_data = global_data[global_data['Type'] == 'confirmed']
            axes[plot_idx].plot(confirmed_data['Date'], confirmed_data['confirmed'], 
                              color=self.colors[0], linewidth=2)
            axes[plot_idx].set_title('Total Confirmed Cases')
            axes[plot_idx].set_ylabel('Cases')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # Plot cumulative deaths
        if 'deaths' in available_types:
            deaths_data = global_data[global_data['Type'] == 'deaths']
            axes[plot_idx].plot(deaths_data['Date'], deaths_data['deaths'], 
                              color=self.colors[1], linewidth=2)
            axes[plot_idx].set_title('Total Deaths')
            axes[plot_idx].set_ylabel('Deaths')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # Plot recovered (if available)
        if 'recovered' in available_types and plot_idx < len(axes):
            recovered_data = global_data[global_data['Type'] == 'recovered']
            axes[plot_idx].plot(recovered_data['Date'], recovered_data['recovered'], 
                              color=self.colors[2], linewidth=2)
            axes[plot_idx].set_title('Total Recovered')
            axes[plot_idx].set_ylabel('Recovered')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # Plot case fatality rate (if both confirmed and deaths are available)
        if 'confirmed' in available_types and 'deaths' in available_types and plot_idx < len(axes):
            confirmed_data = global_data[global_data['Type'] == 'confirmed'][['Date', 'confirmed']]
            deaths_data = global_data[global_data['Type'] == 'deaths'][['Date', 'deaths']]
            
            cfr_data = pd.merge(confirmed_data, deaths_data, on='Date', how='inner')
            if len(cfr_data) > 0:
                cfr_data['CFR'] = (cfr_data['deaths'] / cfr_data['confirmed'].replace(0, np.nan)) * 100
                cfr_data = cfr_data.dropna(subset=['CFR'])
                
                if len(cfr_data) > 0:
                    axes[plot_idx].plot(cfr_data['Date'], cfr_data['CFR'], 
                                      color=self.colors[3], linewidth=2)
                    axes[plot_idx].set_title('Case Fatality Rate (%)')
                    axes[plot_idx].set_ylabel('CFR (%)')
                    axes[plot_idx].tick_params(axis='x', rotation=45)
                    plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_countries(self, processed_data, metric='confirmed', n=10, save_path=None):
        """Plot top N countries by specified metric"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get latest data for each country
        df = processed_data[metric]
        latest_data = df.loc[df.groupby('Country/Region')['Date'].idxmax()]
        top_countries = latest_data.nlargest(n, metric)
        
        # Bar plot
        axes[0].barh(range(len(top_countries)), top_countries[metric], 
                    color=self.colors[:len(top_countries)])
        axes[0].set_yticks(range(len(top_countries)))
        axes[0].set_yticklabels(top_countries['Country/Region'])
        axes[0].set_xlabel(f'Total {metric.capitalize()}')
        axes[0].set_title(f'Top {n} Countries - {metric.capitalize()}')
        
        # Time series for top countries
        top_country_names = top_countries['Country/Region'].tolist()
        for i, country in enumerate(top_country_names[:5]):  # Show top 5 trends
            country_data = df[df['Country/Region'] == country]
            axes[1].plot(country_data['Date'], country_data[metric], 
                        label=country, linewidth=2, color=self.colors[i])
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(f'{metric.capitalize()}')
        axes[1].set_title(f'{metric.capitalize()} Trends - Top 5 Countries')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, owid_data, save_path=None):
        """Plot correlation heatmap of COVID-19 metrics"""
        if owid_data is None:
            print("OWID data not available for correlation analysis")
            return
        
        # Select numeric columns for correlation
        numeric_cols = owid_data.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if 'cases' in col.lower() or 'deaths' in col.lower() or 'vaccin' in col.lower()]
        
        if len(correlation_cols) < 2:
            print("Insufficient numeric columns for correlation analysis")
            return
        
        corr_data = owid_data[correlation_cols].corr()
        
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('COVID-19 Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_vaccination_progress(self, owid_data, countries=None, save_path=None):
        """Plot vaccination progress for selected countries"""
        if owid_data is None or 'people_fully_vaccinated' not in owid_data.columns:
            print("Vaccination data not available")
            return
        
        if countries is None:
            # Select top 10 countries by population
            countries = owid_data.groupby('location')['population'].max().nlargest(10).index.tolist()
        
        plt.figure(figsize=(14, 8))
        
        for i, country in enumerate(countries[:10]):
            country_data = owid_data[owid_data['location'] == country].dropna(subset=['people_fully_vaccinated'])
            if len(country_data) > 0:
                plt.plot(country_data['date'], country_data['people_fully_vaccinated'], 
                        label=country, linewidth=2, color=self.colors[i])
        
        plt.xlabel('Date')
        plt.ylabel('People Fully Vaccinated')
        plt.title('COVID-19 Vaccination Progress by Country')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_dashboard(self, processed_data, owid_data=None, save_path=None):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Global trends
        ax1 = fig.add_subplot(gs[0, :2])
        if 'confirmed' in processed_data:
            global_confirmed = processed_data['confirmed'].groupby('Date')['confirmed'].sum()
            ax1.plot(global_confirmed.index, global_confirmed.values, 
                    color=self.colors[0], linewidth=3)
            ax1.set_title('Global Confirmed Cases Over Time', fontweight='bold')
            ax1.set_ylabel('Confirmed Cases')
        
        # Global deaths
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'deaths' in processed_data:
            global_deaths = processed_data['deaths'].groupby('Date')['deaths'].sum()
            ax2.plot(global_deaths.index, global_deaths.values, 
                    color=self.colors[1], linewidth=3)
            ax2.set_title('Global Deaths Over Time', fontweight='bold')
            ax2.set_ylabel('Deaths')
        
        # Top countries - Cases
        ax3 = fig.add_subplot(gs[1, :2])
        if 'confirmed' in processed_data:
            df = processed_data['confirmed']
            latest_data = df.loc[df.groupby('Country/Region')['Date'].idxmax()]
            top_10 = latest_data.nlargest(10, 'confirmed')
            ax3.barh(range(len(top_10)), top_10['confirmed'], color=self.colors[:len(top_10)])
            ax3.set_yticks(range(len(top_10)))
            ax3.set_yticklabels(top_10['Country/Region'])
            ax3.set_title('Top 10 Countries by Total Cases', fontweight='bold')
        
        # Top countries - Deaths
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'deaths' in processed_data:
            df = processed_data['deaths']
            latest_data = df.loc[df.groupby('Country/Region')['Date'].idxmax()]
            top_10 = latest_data.nlargest(10, 'deaths')
            ax4.barh(range(len(top_10)), top_10['deaths'], color=self.colors[:len(top_10)])
            ax4.set_yticks(range(len(top_10)))
            ax4.set_yticklabels(top_10['Country/Region'])
            ax4.set_title('Top 10 Countries by Total Deaths', fontweight='bold')
        
        # Case fatality rate over time
        ax5 = fig.add_subplot(gs[2, :2])
        if 'confirmed' in processed_data and 'deaths' in processed_data:
            global_conf = processed_data['confirmed'].groupby('Date')['confirmed'].sum()
            global_deaths = processed_data['deaths'].groupby('Date')['deaths'].sum()
            cfr = (global_deaths / global_conf * 100).fillna(0)
            ax5.plot(cfr.index, cfr.values, color=self.colors[3], linewidth=2)
            ax5.set_title('Global Case Fatality Rate Over Time', fontweight='bold')
            ax5.set_ylabel('CFR (%)')
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Calculate summary stats
        if 'confirmed' in processed_data and 'deaths' in processed_data:
            total_cases = processed_data['confirmed'].groupby('Date')['confirmed'].sum().iloc[-1]
            total_deaths = processed_data['deaths'].groupby('Date')['deaths'].sum().iloc[-1]
            cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0
            
            summary_text = f"""
            COVID-19 Global Summary
            
            Total Confirmed Cases: {total_cases:,.0f}
            Total Deaths: {total_deaths:,.0f}
            Case Fatality Rate: {cfr:.2f}%
            Countries Affected: {len(processed_data['confirmed']['Country/Region'].unique())}
            
            Data as of: {processed_data['confirmed']['Date'].max().strftime('%Y-%m-%d')}
            """
            
            ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        fig.suptitle('COVID-19 Global Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()