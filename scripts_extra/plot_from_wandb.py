# %%
import wandb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ==================== CONFIGURATION VARIABLES ====================
# Edit these variables to customize your plotting

# WandB Configuration
WANDB_API_KEY = "71b54366f0dcf364f47a59ed91fd5e5db58a0928"
PROJECT_NAME = "sit_training"
ENTITY_NAME = "tommaso_research"

# Run and Metric Configuration
RUN_NAME = "FINAL2_curriculum_baseline"  # The specific run you want to plot
RUN_ID = 10919215                            # Alternative: use run ID instead of name (set RUN_NAME to None if using this)
                                         # Example: RUN_ID = "87260050" and RUN_NAME = None

# Plot Customization
FIGURE_SIZE = (12, 9)          # Figure size (width, height)
LINE_COLOR = "#1f77b4"         # Nice blue color (matplotlib default)
LINE_WIDTH = 1.5               # Slightly thinner line
GRID = True                    # Whether to show grid
TITLE_FONTSIZE = 16           # Title font size
LABEL_FONTSIZE = 14           # Axis label font size
TICK_FONTSIZE = 12            # Tick label font size

# Style settings for wandb-like appearance
BACKGROUND_COLOR = "#ffffff"   # White background
GRID_COLOR = "#e0e0e0"        # Light gray grid
GRID_ALPHA = 0.3              # Grid transparency
SPINE_COLOR = "#cccccc"       # Axis spine color

# Optional: Smoothing
APPLY_SMOOTHING = True        # Whether to apply smoothing to the data
SMOOTHING_ALPHA = 0.1         # EMA smoothing factor (0.01 = heavy smoothing, 0.9 = light smoothing)

# Optional: X-axis limits (set to None for auto)
X_MIN = None
X_MAX = None

# # Optional: Y-axis limits (set to None for auto)
Y_MIN = None
Y_MAX = None

# Add a dashed vertical line at 20% mark
USE_20_PERCENT_LINE = False

###### INTERACTION TERM PLOT CONFIGURATION ######
# METRIC_NAME = "interaction_term"
# METRIC_DISPLAY_NAME = "Interaction"
# Y_MIN = 0.0007
# Y_MAX = 0.0035
# USE_LOG_SCALE_Y = True
# USE_LOG_SCALE_X = False
# USE_20_PERCENT_LINE = True

##### INTERACTION OVER CONFINEMENT PLOT CONFIGURATION ######
# METRIC_NAME = "interaction_over_confinement"
# METRIC_DISPLAY_NAME = "Interaction / Confinement"
# Y_MIN = 0.02
# Y_MAX = 0.1
# USE_LOG_SCALE_Y = True
# USE_LOG_SCALE_X = False
# USE_20_PERCENT_LINE = True

##### MSE PLOT CONFIGURATION ######
METRIC_NAME = "confinement_term"
METRIC_DISPLAY_NAME = "Mean Squared Error (Confinement Term)"
Y_MIN = 0.03
Y_MAX = 0.2
USE_LOG_SCALE_Y = True
USE_LOG_SCALE_X = False
USE_20_PERCENT_LINE = True
FIGURE_SIZE = (12, 6)

# ==================== FUNCTIONS ====================

def setup_wandb(api_key: str):
    """Initialize wandb API with the provided API key."""
    wandb.login(key=api_key)
    return wandb.Api()

def get_run_data(api, entity: str, project: str, run_name: str = None, run_id: str = None):
    """Retrieve a specific run from wandb by name or ID."""
    try:
        if run_id is not None:
            # Get run by ID
            print(f"Fetching run by ID: {run_id}")
            run = api.run(f"{entity}/{project}/{run_id}")
            return run
        elif run_name is not None:
            # Get run by name
            print(f"Fetching run by name: {run_name}")
            runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
            run_list = list(runs)
            
            if not run_list:
                print(f"No run found with name '{run_name}'. Try using RUN_ID instead.")
                raise ValueError(f"No run found with name '{run_name}'")
            else:
                return run_list[0]
        else:
            raise ValueError("Either run_name or run_id must be provided")
            
    except Exception as e:
        print(f"Error retrieving run: {e}")
        # List available runs for debugging
        print("Available runs (first 10):")
        runs = api.runs(f"{entity}/{project}")
        for run in list(runs)[:10]:  # Show first 10 runs
            print(f"  - Name: {run.name} | ID: {run.id}")
        raise

def extract_metric_data(run, metric_name: str) -> tuple:
    """Extract metric data from a wandb run."""
    history = run.history(samples=999999999)
    
    
    if metric_name not in history.columns:
        print(f"Metric '{metric_name}' not found in run history.")
        print("Available metrics:", list(history.columns))
        raise ValueError(f"Metric '{metric_name}' not found")
    
    # Remove NaN values
    data = history[[metric_name, '_step']].dropna()
    
    steps = data['_step'].values
    values = data[metric_name].values
    
    return steps, values

def apply_ema_smoothing(values: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average smoothing to the values.
    
    Args:
        values: Array of values to smooth
        alpha: Smoothing factor (0 < alpha <= 1). Lower values = more smoothing.
               Typical values: 0.01-0.1 (heavy smoothing), 0.3-0.6 (moderate), 0.7-0.9 (light)
    
    Returns:
        Smoothed values array
    """
    if len(values) == 0:
        return values
    
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]  # First value remains unchanged
    
    # Simple EMA without bias correction to avoid potential issues
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    
    return smoothed

def format_thousands(x, pos):
    """Formatter function for x-axis to add comma separators."""
    if x >= 1000:
        return f'{int(x):,}'
    else:
        return f'{int(x)}'

def format_decimal(x, pos):
    """Formatter function for y-axis to show decimal numbers instead of scientific notation."""
    # Get the current axis to check if this is the lowest tick
    import matplotlib.pyplot as plt
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    
    # Don't show label for the lowest tick (within a small tolerance)
    if abs(x - y_min) < (y_max - y_min) * 0.01:
        return ''
    
    if x == 0:
        return '0'
    elif x >= 1:
        return f'{x:.1f}'
    elif x >= 0.1:
        return f'{x:.2f}'
    elif x >= 0.01:
        return f'{x:.3f}'
    elif x >= 0.001:
        return f'{x:.4f}'
    else:
        # For very small numbers, still use decimal notation
        return f'{x:.5f}'

def format_percentage(x, pos, max_steps):
    """Formatter function for x-axis to show percentages."""
    percentage = (x / max_steps) * 100
    return f'{int(percentage)}%'

def create_plot(steps: np.ndarray, values: np.ndarray, 
                metric_display_name: str, run_name: str) -> plt.Figure:
    """Create a matplotlib plot of the metric data."""
    
    # Store original values for background plotting if smoothing is applied
    original_values = values.copy()
    
    # Apply smoothing if requested
    if APPLY_SMOOTHING:
        values = apply_ema_smoothing(values, SMOOTHING_ALPHA)
    
    # Set up the style to look like wandb UI
    plt.style.use('default')  # Start with clean default style
    
    # Create the plot with wandb-like styling
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Plot the original data in light blue if smoothing is applied (like wandb UI)
    if APPLY_SMOOTHING:
        ax.plot(steps, original_values, color=LINE_COLOR, linewidth=0.5, alpha=0.3, zorder=1)
    
    # Plot the main data (smoothed or original) with wandb-like styling
    ax.plot(steps, values, color=LINE_COLOR, linewidth=LINE_WIDTH, alpha=0.8, zorder=2)
    
    # Customize the plot to look like wandb UI
    ax.set_xlabel('')  # Remove x-axis label, we'll add it inside the plot
    ax.set_ylabel('')  # Remove y-axis label as requested
    ax.set_title(f'{metric_display_name}', 
                fontsize=TITLE_FONTSIZE, color='#333333', pad=20)
    
    # Set log scale FIRST if requested
    if USE_LOG_SCALE_Y:
        ax.set_yscale('log')
    if USE_LOG_SCALE_X:
        ax.set_xscale('log')
    
    # Set axis limits AFTER log scale (this ensures limits work correctly)
    if X_MIN is not None or X_MAX is not None:
        ax.set_xlim(X_MIN, X_MAX)
    if Y_MIN is not None or Y_MAX is not None:
        ax.set_ylim(Y_MIN, Y_MAX)
    
    # Style the axes and ticks like wandb
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE, colors='#666666')
    ax.tick_params(axis='both', which='major', length=0)  # Remove tick marks
    ax.tick_params(axis='x', pad=8)  # Add spacing between x-axis tick text and plot
    ax.tick_params(axis='y', pad=8)  # Add spacing between y-axis tick text and plot
    
    # Format y-axis to show decimal numbers instead of scientific notation
    y_formatter = FuncFormatter(format_decimal)
    ax.yaxis.set_major_formatter(y_formatter)
    
    # For log scale, also set minor formatter and ensure proper tick locations
    if USE_LOG_SCALE_Y:
        # Don't use minor formatter for log scale to avoid conflicts
        # Force specific tick locations to avoid scientific notation
        from matplotlib.ticker import FixedLocator
        
        # Get the current y-axis limits to determine appropriate tick positions
        y_min, y_max = ax.get_ylim()
        
        # Create specific tick positions
        import numpy as np
        tick_values = []
        
        # Generate tick values in the visible range
        power = int(np.floor(np.log10(y_min)))
        while 10**power <= y_max * 1.1:
            base = 10**power
            for mult in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                val = mult * base
                if y_min <= val <= y_max * 1.1:
                    tick_values.append(val)
            power += 1
        
        # Set the tick locations explicitly
        ax.yaxis.set_major_locator(FixedLocator(tick_values))
        ax.yaxis.set_major_formatter(y_formatter)
    
    # Style the spines (borders)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)
    
    # Add grid with wandb-like styling
    if GRID:
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.8)
        ax.set_axisbelow(True)  # Put grid behind the plot
    
    # Format the x-axis
    if len(steps) > 0:
        if USE_LOG_SCALE_X:
            # For log scale, use standard numerical formatting
            formatter = FuncFormatter(format_thousands)
            ax.xaxis.set_major_formatter(formatter)
        else:
            # For linear scale, use percentages (0% to 100%)
            max_steps = steps.max()
            min_steps = steps.min()
            
            # Create a partial function with max_steps bound
            from functools import partial
            percentage_formatter = partial(format_percentage, max_steps=max_steps)
            formatter = FuncFormatter(percentage_formatter)
            ax.xaxis.set_major_formatter(formatter)
            
            # Set nice tick locations at 0%, 10%, 20%, ..., 100%
            tick_positions = [max_steps * i / 10 for i in range(11)]  # 0, 10%, 20%, ..., 100%
            ax.set_xticks(tick_positions)
            
            # Set x-axis limits with a tiny margin on the left (0%)
            margin = (max_steps - min_steps) * 0.005  # 1% margin
            ax.set_xlim(min_steps - margin, max_steps)
            
            # Add dashed black vertical line at 20% mark
            if USE_20_PERCENT_LINE:
                x_20_percent = max_steps * 0.2
                ax.axvline(x=x_20_percent, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
    
    # Add "Step" label inside the plot (bottom right)
    ax.text(0.98, 0.02, 'Step', transform=ax.transAxes, 
            fontsize=LABEL_FONTSIZE, color='#666666', 
            horizontalalignment='right', verticalalignment='bottom')
    
    # Remove the legend since it's just one line and the title explains it
    # This makes it cleaner like wandb UI
    
    plt.tight_layout()
    
    return fig

def save_plot(fig: plt.Figure, run_name: str, metric_name: str, 
              output_dir: str = "plots") -> str:
    """Save the plot to a file."""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"{run_name}_{metric_name}_plot.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    return filepath

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to execute the plotting pipeline."""
    
    # Setup wandb API
    print("Setting up wandb API...")
    api = setup_wandb(WANDB_API_KEY)
    
    # Get the run data
    if RUN_ID is not None:
        print(f"Retrieving run by ID '{RUN_ID}' from project '{PROJECT_NAME}'...")
        run = get_run_data(api, ENTITY_NAME, PROJECT_NAME, run_id=RUN_ID)
    elif RUN_NAME is not None:
        print(f"Retrieving run by name '{RUN_NAME}' from project '{PROJECT_NAME}'...")
        run = get_run_data(api, ENTITY_NAME, PROJECT_NAME, run_name=RUN_NAME)
    else:
        raise ValueError("Either RUN_NAME or RUN_ID must be specified in the configuration")
    
    print(f"Found run: {run.name} (ID: {run.id})")
    
    # Extract metric data
    print(f"Extracting metric '{METRIC_NAME}'...")
    steps, values = extract_metric_data(run, METRIC_NAME)
    print(f"Found {len(steps)} data points")
    print(f"Value range: {values.min():.6f} to {values.max():.6f}")
    print(f"Step range: {steps.min()} to {steps.max()}")
    
    # Create the plot
    print("Creating plot...")
    fig = create_plot(steps, values, METRIC_DISPLAY_NAME, RUN_NAME)
    
    # Save the plot
    plot_run_name = RUN_NAME if RUN_NAME is not None else RUN_ID
    save_plot(fig, plot_run_name, METRIC_NAME)
    
    # Try to show the plot (may not work in headless environments)
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot (this is normal in headless environments): {e}")
        print("Plot has been saved to file instead.")
    
    print("Done!")

if __name__ == "__main__":
    main()