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
            cumulative_df = pd.concat([cumulative_df, filtered_df], ignore_index=True)
        else:
            print(f"CSV file not found: {csv_path}")
        
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Group by mode, algorithm, and cfg_scale to create separate lines
    for (sampling, exp_name), group in cumulative_df.groupby(['sampling', 'exp_name']):
        group_sorted = group.sort_values('steps')
        plt.plot(group_sorted['steps'], group_sorted['fid'], 
                marker='o', label=f'{exp_name} - {sampling}')

    plt.xlabel('Sampling Steps')
    plt.ylabel('FID Score')
    plt.title('FID Score vs Sampling Steps')
    plt.yticks(range(0, int(cumulative_df['fid'].max()) + 1, 10))
    plt.xscale('log')
    plt.xticks(args.sampling_steps, [str(step) for step in args.sampling_steps])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # Save the plot
    plot_path = os.path.join(args.plot_out, 'fid_plot.png')
    os.makedirs(args.plot_out, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    
def create_argparser():
    defaults = dict(
        exp_name=[""],
        sampling_steps=[3, 4, 5, 10, 30, 50, 100],
        sampling_modes=["DDIM"],
        # cfg_scales=[0.0, 1.0, 2.0],  # Uncomment if cfg scales are needed
        samples_dir=[""],
        plot_out="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # main(create_argparser().parse_args())
    exp_name = [
        "cifar10_uncond_openai",
        "cifar10_cond_openai",
        "cifar10_cond_baseline",
        "cifar10_cond_distributional_logsnr",
        "cifar10_cond_distributional_noweighting",
        "cifar10_cond_distributional_noweighting_lambda_linear",
    ]
    
    args = EasyDict(dict(
        exp_name=exp_name,
        sampling_steps=[3, 4, 5, 10, 30, 50, 100],
        sampling_modes=["DDIM"],
        # cfg_scales=[0.0, 1.0, 2.0],  # Uncomment if cfg scales are needed
        samples_dir=[
            f"/ceph/scratch/martorellat/guided_diffusion/samples_{name}"
            for name in exp_name
        ],
        plot_out="/ceph/scratch/martorellat/guided_diffusion/figures",
    ))
    main(args)