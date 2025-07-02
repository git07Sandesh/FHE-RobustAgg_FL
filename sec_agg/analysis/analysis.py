# --- START OF FILE analysis.py ---

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# IMPORTANT: Change this to your W&B project details
WANDB_PROJECT_PATH = "hamza-latif/fl-robustness-evaluation" # <--- CHANGE THIS
# Directory to save the plots
SAVE_DIR = "analysis_plots"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def fetch_wandb_data(project_path: str) -> pd.DataFrame:
    """Fetches run data from W&B and returns it as a pandas DataFrame."""
    print(f"Fetching data from W&B project: {project_path}")
    api = wandb.Api()
    runs = api.runs(path=project_path)

    summary_list = []
    for run in runs:
        if run.state != "finished":
            print(f"Skipping unfinished run: {run.name}")
            continue
        
        summary = {**run.summary._json_dict}
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        # Combine config and summary, giving summary precedence for overlapping keys
        run_data = {**config, **summary}
        summary_list.append(run_data)

    df = pd.DataFrame(summary_list)
    print(f"Successfully fetched {len(df)} finished runs.")
    return df

def plot_cost_of_robustness(df: pd.DataFrame):
    """Generates a bar chart for the 'Cost of Robustness' ratio."""
    print("\nGenerating 'Cost of Robustness' plot...")
    
    # Filter for the main attack scenario (n=10, f=3)
    attack_df = df[(df['num_partitions'] == 10) & (df['num_malicious'] == 3)].copy()
    
    fedavg_run = attack_df[attack_df['strategy'] == 'FedAvg']
    if fedavg_run.empty:
        print("Could not find FedAvg baseline for n=10, f=3. Skipping plot.")
        return

    base_agg_time = fedavg_run['final_total_aggregation_time'].iloc[0]
    
    # Calculate model size from a fedavg run to compute communication cost ratio
    # Assuming uplink and downlink are roughly the same for one model
    base_comm_cost_mb = df[df['run_name'] == 'fedavg_benign_f0_n10']['downlink_mb'].iloc[0]

    robust_df = attack_df[attack_df['strategy'] != 'FedAvg']
    if robust_df.empty:
        print("No robust strategies found for n=10, f=3. Skipping plot.")
        return

    robust_df['Comp_Cost_Ratio'] = robust_df['final_total_aggregation_time'] / base_agg_time
    # For now, we assume plaintext communication cost is similar across strategies
    # This would change for HE schemes
    robust_df['Comm_Cost_Ratio'] = robust_df['downlink_mb'] / base_comm_cost_mb

    plot_df = robust_df[['strategy', 'Comp_Cost_Ratio', 'Comm_Cost_Ratio']].melt(
        id_vars='strategy', var_name='Cost Type', value_name='Ratio'
    )
    plot_df['Cost Type'] = plot_df['Cost Type'].replace({
        'Comp_Cost_Ratio': 'Computational Cost',
        'Comm_Cost_Ratio': 'Communication Cost'
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='strategy', y='Ratio', hue='Cost Type')
    plt.title('Relative Cost of Robustness vs. FedAvg (n=10, f=3)')
    plt.ylabel('Cost Ratio (Higher is more expensive)')
    plt.xlabel('Robust Aggregation Strategy')
    plt.axhline(1.0, color='r', linestyle='--', label='FedAvg Baseline')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "cost_of_robustness.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_accuracy_vs_cost(df: pd.DataFrame):
    """Generates a scatter plot for the 'Accuracy vs. Cost' trade-off."""
    print("\nGenerating 'Accuracy vs. Cost' plot...")
    
    plot_df = df.copy()
    plot_df['cost'] = plot_df['final_total_aggregation_time']
    plot_df['performance'] = plot_df['final_best_accuracy']

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_df, x='cost', y='performance', hue='strategy', style='attack_type', s=150, alpha=0.8)
    
    # Annotate points
    for i, row in plot_df.iterrows():
        plt.text(row['cost'] * 1.01, row['performance'], row['run_name'], fontsize=8)

    plt.title('Accuracy vs. Computational Cost Trade-off')
    plt.xlabel('Total Aggregation Time (seconds) --> More Expensive')
    plt.ylabel('Best Accuracy --> More Effective')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "accuracy_vs_cost.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_scalability(df: pd.DataFrame):
    """Plots aggregation time vs. number of clients for FedAvg and Bulyan."""
    print("\nGenerating 'Scalability Analysis' plot...")
    
    # Filter for runs with varying num_partitions and ~30% malicious clients
    scalability_df = df[df['run_name'].str.contains('fedavg_attack_f|bulyan_attack_f')].copy()
    scalability_df = scalability_df[scalability_df['strategy'].isin(['FedAvg', 'Bulyan'])]
    
    if scalability_df.empty:
        print("No scalability experiment data found. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scalability_df, x='num_partitions', y='final_total_aggregation_time', hue='strategy', marker='o')
    plt.title('Scalability Analysis: Aggregation Time vs. Number of Clients')
    plt.xlabel('Number of Clients (n)')
    plt.ylabel('Total Aggregation Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "scalability_analysis.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_break_point(df: pd.DataFrame):
    """Plots best accuracy vs. the ratio of malicious clients."""
    print("\nGenerating 'Break-Point Analysis' plot...")

    # Filter for break-point experiments (fixed n=10, varying f)
    break_point_df = df[df['num_partitions'] == 10].copy()
    break_point_df = break_point_df[break_point_df['strategy'].isin(['FedAvg', 'TrimmedMean', 'Bulyan'])]
    break_point_df = break_point_df[break_point_df['num_malicious'] > 0]
    
    if break_point_df.empty:
        print("No break-point experiment data found. Skipping plot.")
        return
        
    break_point_df['malicious_ratio'] = break_point_df['num_malicious'] / break_point_df['num_partitions']

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=break_point_df, x='malicious_ratio', y='final_best_accuracy', hue='strategy', marker='o')
    plt.title('Break-Point Analysis: Accuracy vs. Malicious Client Ratio (n=10)')
    plt.xlabel('Malicious Client Ratio (f/n)')
    plt.ylabel('Best Accuracy Achieved')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "break_point_analysis.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    try:
        data = fetch_wandb_data(WANDB_PROJECT_PATH)
        if not data.empty:
            plot_cost_of_robustness(data)
            plot_accuracy_vs_cost(data)
            plot_scalability(data)
            plot_break_point(data)
            print("\n✅ Analysis complete. Plots are saved in the 'analysis_plots' directory.")
        else:
            print("\nNo data found. Please run experiments first.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have run 'wandb login' and set the correct WANDB_PROJECT_PATH.")