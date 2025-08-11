
#%%
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from guided_diffusion.script_util import add_dict_to_argparser

class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

# def get_percentage_from_name(name):
#     if "baseline" in name:
#         return 0
#     parts = name.split("_")
#     for part in parts:
#         if "percent" in part:
#             start_percentage = int(part.replace("percent", "").replace("at", ""))
#             return 100 - start_percentage
#     return None
    
def get_percentage_from_name(name):    
    if "baseline" in name:
        return 100
    parts = name.split("_")
    for part in parts:
        if "percent" in part:
            start_percentage = int(part.replace("percent", "").replace("at", ""))
            return start_percentage
    return None

def main(args):
    
    # csv keys steps,sampling,inception_score,fid,sfid,precision,recall,timestamp
    
    cumulative_df = pd.DataFrame()

    for samples_dir, exp_name in zip(args.samples_dir, args.exp_name):
        print(f"Processing samples directory: {samples_dir} for experiment: {exp_name}")
        csv_path = os.path.join(samples_dir, "results.csv")
        # Load and filter CSV data
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Filter data based on provided parameters
            filtered_df = df[
                (df['steps'].isin(args.sampling_steps)) &
                # (df['cfg'].isin(args.cfg_scales)) &
                (df['sampling'].isin(args.sampling_modes)) &
                (df['num_samples'] == 50000)  # Assuming 50000 samples for FID
            ]
            
            # Take last timestamped row for each configuration
            filtered_df = filtered_df.sort_values('timestamp').drop_duplicates(
                subset=['steps', 'sampling'], keep='last'
            )
            
            # Append to cumulative DataFrame
            filtered_df['exp_name'] = exp_name
            filtered_df['percentage'] = get_percentage_from_name(exp_name)
            cumulative_df = pd.concat([cumulative_df, filtered_df], ignore_index=True)
        else:
            print(f"CSV file not found: {csv_path}")
        
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.4)
    
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 'x', 's', 'D', '^']
    
    # Define which direction is better for each metric
    better_direction = {
        'fid': '↓',  # Lower is better
        'sfid': '↓',  # Lower is better
        'inception_score': '↑',  # Higher is better
        'precision': '↑',  # Higher is better
        'recall': '↑'  # Higher is better
    }
    metric_pretty_name = {
        'FID': 'FID',
        'SFID': 'sFID',
        'INCEPTION_SCORE': 'Inception Score',
        'PRECISION': 'Precision',
        'RECALL': 'Recall'
    }
    
    # Separate metrics by optimization direction
    higher_better = []
    lower_better = []
    
    for metric in args.metrics:
        if better_direction.get(metric.lower(), '') == '↑':
            higher_better.append(metric)
        else:
            lower_better.append(metric)
            
    print("culmulative_df", cumulative_df)
    
    group_sorted = cumulative_df.sort_values('percentage')
    percentages = sorted(cumulative_df['percentage'].unique())
    
    # Plot metrics where higher is better (left subplot)
    if higher_better:
        axes_higher = [ax1]
        
        # Calculate shared y-bounds for precision and recall
        precision_recall_metrics = [m for m in higher_better if m.lower() in ['precision', 'recall']]
        if len(precision_recall_metrics) > 1:
            all_values = []
            for metric in precision_recall_metrics:
                all_values.extend(group_sorted[metric].values)
            shared_min = min(all_values) * 0.95  # Add 5% padding
            shared_max = max(all_values) * 1.05  # Add 5% padding
        
        for i, metric in enumerate(higher_better):
            if i >= len(colors):
                continue
            
            if i == 0:
                current_ax = ax1
            else:
                current_ax = ax1.twinx()
                axes_higher.append(current_ax)
                # Position additional y-axes on the right side for this subplot
                current_ax.spines['right'].set_position(('outward', 60 * (i - 1)))
            
            color = colors[i]
            marker = markers[i]
            
            current_ax.plot(group_sorted['percentage'], group_sorted[metric], 
                           marker=marker, color=color, linewidth=2)
            current_ax.set_ylabel(f'{metric_pretty_name[metric.upper()]} ↑', color=color)
            current_ax.tick_params(axis='y', labelcolor=color)
            
            # Set specific y-axis bounds
            if metric.lower() in ['precision', 'recall'] and len(precision_recall_metrics) > 1:
                current_ax.set_ylim(shared_min, shared_max)
        
        ax1.axvline(x=percentages[2], color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
        ax1.set_xlabel('Percentage of Training using Standard Regression Loss')
        ax1.set_title('Higher is Better ↑')
        ax1.set_xticks(percentages)
        ax1.set_xticklabels([f'{p}%' for p in percentages])
        # ax1.invert_xaxis()  # Reverse the x-axis
        ax1.grid(True, alpha=0.3)
    
    # Plot metrics where lower is better (right subplot)
    if lower_better:
        axes_lower = [ax2]
        # Use different colors for lower_better metrics to avoid red/blue from left plot
        lower_colors = ['tab:orange', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive']
        
        for i, metric in enumerate(lower_better):
            if i >= len(lower_colors):
                continue
            
            if i == 0:
                current_ax = ax2
            else:
                current_ax = ax2.twinx()
                axes_lower.append(current_ax)
                # Position additional y-axes on the right side for this subplot
                current_ax.spines['right'].set_position(('outward', 60 * (i - 1)))
            
            color = lower_colors[i]
            marker = markers[i]
            
            current_ax.plot(group_sorted['percentage'], group_sorted[metric], 
                           marker=marker, color=color, linewidth=2)
            current_ax.set_ylabel(f'{metric_pretty_name[metric.upper()]} ↓', color=color)
            current_ax.tick_params(axis='y', labelcolor=color)
            
            # Set specific y-axis bounds for FID and sFID
            if metric.lower() in ['fid', 'sfid']:
                current_ax.set_ylim(4, 15)
        
        ax2.axvline(x=percentages[2], color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
        ax2.set_xlabel('Percentage of Training using Standard Regression Loss')
        ax2.set_title('Lower is Better ↓')
        ax2.set_xticks(percentages)
        ax2.set_xticklabels([f'{p}%' for p in percentages])
        # ax2.invert_xaxis()  # Reverse the x-axis
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Metrics vs Checkpoint of Switch to Distributional Training ({args.sampling_steps[0]} steps)', fontsize=16)
    
    # Save the plot
    metrics_str = "_vs_".join(args.metrics)
    plot_path = os.path.join(args.plot_out, f'curriculum_{metrics_str}_plot.png')
    os.makedirs(args.plot_out, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {plot_path}")
    
    plt.show()
    
    
def create_argparser():
    defaults = dict(
        exp_name=[""],
        sampling_steps=[100],
        sampling_modes=["DDIM"],
        metrics=["fid", "inception_score"],
        # cfg_scales=[0.0, 1.0, 2.0],  # Uncomment if cfg scales are needed
        samples_dir=[""],
        plot_out="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    exp_name = [
        "FINAL_curriculum_baseline",
        "FINAL3_curriculum_start_at_00percent",
        "curriculum_start_at_10percent",
        "curriculum_start_at_20percent",
        "curriculum_start_at_30percent",
        "curriculum_start_at_40percent",
        "curriculum_start_at_50percent",
        "curriculum_start_at_60percent",
        "curriculum_start_at_70percent",
        "curriculum_start_at_80percent",
        "curriculum_start_at_90percent",
    ]
    
    args = EasyDict(dict(
        exp_name=exp_name,
        sampling_steps=[50],
        sampling_modes=["DDIM"],
        metrics=["fid", "sfid", "inception_score", "precision", "recall"],
        samples_dir=[
            f"/ceph/scratch/martorellat/guided_diffusion/curriculum/samples_{name}"
            for name in exp_name
        ],
        plot_out="/ceph/scratch/martorellat/guided_diffusion/figures",
    ))
    main(args)
    # main(create_argparser().parse_args())
# %%
