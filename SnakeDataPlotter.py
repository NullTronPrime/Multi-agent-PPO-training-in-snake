import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import argparse
from datetime import datetime

def find_run_folders(base_dir='.'):
    """Find all run folders in the given directory."""
    return [d for d in glob.glob(os.path.join(base_dir, "run_*")) if os.path.isdir(d)]

def load_run_data(run_folder):
    """Load training data from a run folder."""
    csv_path = os.path.join(run_folder, "training_details.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: No training data found in {run_folder}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Add run folder name as identifier
        run_name = os.path.basename(run_folder)
        df['run_name'] = run_name
        return df
    except Exception as e:
        print(f"Error loading data from {run_folder}: {e}")
        return None

def generate_comparative_plots(run_data_list, output_dir=None, smoothing=5):
    """Generate comparative plots for all runs."""
    if not run_data_list:
        print("No valid run data to plot.")
        return
    
    # Create a timestamp-based output directory if none provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"snake_plots_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a color palette for the runs
    n_runs = len(run_data_list)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_runs))
    
    # Get relevant columns to plot (excluding non-numeric or identifier columns)
    sample_df = run_data_list[0]
    plot_columns = [col for col in sample_df.columns if col not in ['run_name', 'episode_count']]
    
    # Create individual plots for each metric
    for col in plot_columns:
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        for i, df in enumerate(run_data_list):
            run_name = df['run_name'].iloc[0]
            
            # Apply smoothing if specified
            if smoothing > 1:
                y = df[col].rolling(window=smoothing).mean()
                # Fill NaN values at the beginning due to the rolling window
                y = y.fillna(df[col].iloc[:smoothing].mean())
            else:
                y = df[col]
            
            ax.plot(df['update'], y, label=run_name, color=colors[i], linewidth=2)
            
            # Add final value annotation
            final_x = df['update'].iloc[-1]
            final_y = y.iloc[-1]
            ax.annotate(f"{final_y:.2f}", 
                        xy=(final_x, final_y),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[i])
        
        # Formatting
        ax.set_title(f"{col} across Training Runs", fontsize=16)
        ax.set_xlabel("Update", fontsize=14)
        ax.set_ylabel(col, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Improve legend placement and readability
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=min(5, n_runs), frameon=True, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate episode count plot if available
    if 'episode_count' in sample_df.columns:
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        for i, df in enumerate(run_data_list):
            run_name = df['run_name'].iloc[0]
            ax.plot(df['update'], df['episode_count'], label=run_name, color=colors[i], linewidth=2)
            
            # Add final count annotation
            final_x = df['update'].iloc[-1]
            final_y = df['episode_count'].iloc[-1]
            ax.annotate(f"{final_y}", 
                        xy=(final_x, final_y),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[i])
        
        ax.set_title("Episodes Completed across Training Runs", fontsize=16)
        ax.set_xlabel("Update", fontsize=14)
        ax.set_ylabel("Episodes", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=min(5, n_runs), frameon=True, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "episode_count_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined learning curve plot (avg_reward and best_reward)
    if 'avg_reward' in sample_df.columns and 'best_reward' in sample_df.columns:
        plt.figure(figsize=(14, 10))
        
        # Top subplot for average reward
        ax1 = plt.subplot(211)
        for i, df in enumerate(run_data_list):
            run_name = df['run_name'].iloc[0]
            
            if smoothing > 1:
                y = df['avg_reward'].rolling(window=smoothing).mean()
                y = y.fillna(df['avg_reward'].iloc[:smoothing].mean())
            else:
                y = df['avg_reward']
                
            ax1.plot(df['update'], y, label=run_name, color=colors[i], linewidth=2)
        
        ax1.set_title("Average Reward across Training Runs", fontsize=16)
        ax1.set_ylabel("Average Reward", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Bottom subplot for best reward
        ax2 = plt.subplot(212, sharex=ax1)
        for i, df in enumerate(run_data_list):
            run_name = df['run_name'].iloc[0]
            ax2.plot(df['update'], df['best_reward'], label=run_name, color=colors[i], linewidth=2)
            
            # Add final value annotation
            final_x = df['update'].iloc[-1]
            final_y = df['best_reward'].iloc[-1]
            ax2.annotate(f"{final_y:.2f}", 
                        xy=(final_x, final_y),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[i])
        
        ax2.set_title("Best Reward across Training Runs", fontsize=16)
        ax2.set_xlabel("Update", fontsize=14)
        ax2.set_ylabel("Best Reward", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Improve legend placement and readability (only on bottom plot)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  ncol=min(5, n_runs), frameon=True, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_curves_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create update duration plot if available
    if 'update_duration' in sample_df.columns:
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        for i, df in enumerate(run_data_list):
            run_name = df['run_name'].iloc[0]
            
            if smoothing > 1:
                y = df['update_duration'].rolling(window=smoothing).mean()
                y = y.fillna(df['update_duration'].iloc[:smoothing].mean())
            else:
                y = df['update_duration']
                
            ax.plot(df['update'], y, label=run_name, color=colors[i], linewidth=2)
            
            # Calculate and annotate average duration
            avg_duration = df['update_duration'].mean()
            ax.annotate(f"Avg: {avg_duration:.2f}s", 
                        xy=(df['update'].iloc[-1], y.iloc[-1]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[i])
        
        ax.set_title("Update Duration across Training Runs", fontsize=16)
        ax.set_xlabel("Update", fontsize=14)
        ax.set_ylabel("Duration (seconds)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=min(5, n_runs), frameon=True, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "update_duration_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to directory: {output_dir}")
    
    # Generate a summary statistics table
    summary_stats = []
    for df in run_data_list:
        run_name = df['run_name'].iloc[0]
        run_stats = {
            'Run': run_name,
            'Updates': df['update'].max() + 1,
            'Episodes': df['episode_count'].max() if 'episode_count' in df.columns else 'N/A',
            'Final Avg Reward': df['avg_reward'].iloc[-1] if 'avg_reward' in df.columns else 'N/A',
            'Best Reward': df['best_reward'].max() if 'best_reward' in df.columns else 'N/A',
            'Avg Update Duration': df['update_duration'].mean() if 'update_duration' in df.columns else 'N/A'
        }
        summary_stats.append(run_stats)
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, "run_summary.csv"), index=False)
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Generate comparative plots for Snake PPO training runs')
    parser.add_argument('--base_dir', default='.', help='Base directory containing run folders')
    parser.add_argument('--output_dir', default=None, help='Output directory for plots')
    parser.add_argument('--smoothing', type=int, default=5, help='Smoothing window size for metrics')
    parser.add_argument('--specific_runs', nargs='*', help='Specific run folders to analyze')
    args = parser.parse_args()
    
    print(f"Looking for run folders in {args.base_dir}")
    
    if args.specific_runs:
        run_folders = [os.path.join(args.base_dir, run) for run in args.specific_runs]
        print(f"Analyzing specific runs: {args.specific_runs}")
    else:
        run_folders = find_run_folders(args.base_dir)
        print(f"Found {len(run_folders)} run folders")
    
    run_data_list = []
    for folder in run_folders:
        print(f"Loading data from {folder}")
        data = load_run_data(folder)
        if data is not None:
            run_data_list.append(data)
    
    output_dir = generate_comparative_plots(run_data_list, args.output_dir, args.smoothing)
    
    if output_dir:
        print(f"Analysis complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
