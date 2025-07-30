#%%
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
# from torchvision import transforms
from tqdm import tqdm


# Assuming these are available in your project structure
from data_utils import load_data
from guided_diffusion.script_util import create_gaussian_diffusion

def exists(val):
	return val is not None


def is_lambda(f):
	return callable(f)  # and f.__name__ == "<lambda>"


def default(val, d, eager=True):
	if exists(val):
		return val
	return d() if is_lambda(d) and eager else d
	
def right_pad_dims_to(x, t):
	padding_dims = x.ndim - t.ndim
	if padding_dims <= 0:
		return t
	return t.view(*t.shape, *((1,) * padding_dims))


def estimate_conditional_variance(
	x0_samples: torch.Tensor, xt: torch.Tensor, p_xt_given_x0_log_prob: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Estimates the conditional mean and variance E[X₀|Xₜ] and Var(X₀|Xₜ)
	using self-normalized importance sampling with robust numerical stability
	for high-dimensional data.
	"""
	# Ensure inputs are in the correct format
	if xt.dim() == 1:
		xt = xt.unsqueeze(0)
	if x0_samples.dim() == 1:
		x0_samples = x0_samples.unsqueeze(0)

	B, D = xt.shape
	N, _ = x0_samples.shape

	# Step 1: Calculate the log importance weights log p(xₜ|x₀)
	log_weights = p_xt_given_x0_log_prob(xt, x0_samples)  # Shape (B, N)

	# --- NUMERICAL STABILITY FIX ---
	# Step 2: Shift the log weights by subtracting the maximum value in each row.
	# This prevents underflow when we exponentiate.
	max_log_weights, _ = torch.max(log_weights, dim=1, keepdim=True)

	# Check for cases where all log_weights are -inf (a real possibility)
	# If so, the weights are undefined, we can return zero variance as a fallback.
	if torch.isinf(max_log_weights).any():
		print("Warning: All log weights are -inf. Weights are ill-defined. Returning zero variance.")
		# Create zero tensors with the correct shape and device
		zero_mean = torch.zeros_like(xt)
		zero_var = torch.zeros_like(xt)
		return zero_mean, zero_var

	shifted_log_weights = log_weights - max_log_weights

	# Step 3: Calculate weights in linear space. No underflow will occur here.
	weights = torch.exp(shifted_log_weights)

	# Step 4: Normalize the weights.
	sum_weights = torch.sum(weights, dim=1, keepdim=True)
	normalized_weights = weights / sum_weights
	# --- END OF FIX ---

	# Add dimensions for broadcasting:
	weights_broadcast = normalized_weights.unsqueeze(2)
	x0_broadcast = x0_samples.unsqueeze(0)

	# Estimate the conditional mean
	estimated_mean = torch.sum(weights_broadcast * x0_broadcast, dim=1)

	# Estimate the conditional second moment
	estimated_second_moment = torch.sum(weights_broadcast * x0_broadcast.pow(2), dim=1)

	# Calculate the conditional variance
	estimated_variance = estimated_second_moment - estimated_mean.pow(2)

	# Clamp to prevent tiny negative values from floating point inaccuracies
	estimated_variance.clamp_(min=1e-9)

	return estimated_mean, estimated_variance


def create_log_prob_function_optimized(diffusion, t_value, x0_importance_norms_sq, D):
	"""
	Creates an optimized p_xt_given_x0_log_prob function that uses matrix
	multiplication and a pre-computed (cached) component.
	"""
	# Using continuous time, t_value is a float from 0 to 1
	t_tensor = torch.tensor([t_value], device=x0_importance_norms_sq.device)  # Or your device
	# schedule_t = noise_schedule(t_tensor)
	# alpha_t = schedule_t["alpha"].item()
	# sigma_t = schedule_t["sigma"].item()
	alpha_t = diffusion.sqrt_alphas_cumprod[t_value]
	sigma_t = diffusion.sqrt_one_minus_alphas_cumprod[t_value]

	# Pre-calculate constant part of the log-pdf
	# *** FIX: Remove the incorrect line. D is now passed in correctly. ***
	# D = x0_importance_norms_sq.shape[1] # <--- REMOVED
	const_term = -(D / 2.0) * math.log(2 * math.pi) - D * math.log(sigma_t)
	inv_two_sigma_sq = 1.0 / (2 * sigma_t**2)

	def p_xt_given_x0_log_prob_optimized(xt: torch.Tensor, x0_samples: torch.Tensor) -> torch.Tensor:
		"""
		Calculates log p(xt | x0) using the GEMM-based approach.
		xt shape: (B, D), x0_samples shape: (M, D)
		"""
		# Calculate terms of the expanded squared norm
		xt_norm_sq = torch.sum(xt.pow(2), dim=1, keepdim=True)  # Shape (B, 1)

		# This is the main matrix multiplication
		dot_products = xt @ x0_samples.T  # Shape (B, M)

		# Combine everything using broadcasting
		# Shapes: (B, 1) - 2*alpha*(B, M) + alpha^2*(1, M) -> (B, M)
		sq_distances = xt_norm_sq - 2 * alpha_t * dot_products + (alpha_t**2) * x0_importance_norms_sq.unsqueeze(0)

		log_prob = const_term - inv_two_sigma_sq * sq_distances
		return log_prob  # Final shape: (B, M)

	return p_xt_given_x0_log_prob_optimized



def main():
	"""
	Calculates the irreducible variance for a diffusion model setup and plots
	the variance as a function of the diffusion time t.
	"""
	# --- 1. Configuration ---
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {DEVICE}")

	N_BENCHMARK = 25000
	M_IMPORTANCE = 25000
	TOTAL_SAMPLES = N_BENCHMARK + M_IMPORTANCE
	NUM_T_BINS = 50
	
	USE_LOSS_WEIGHTING = False
	dataset_name = "cifar10"

	loss_b = 0
	loss_weighting_fn = lambda logsnr: (1 + (loss_b - logsnr).exp()) ** -1

	data = load_data(
        data_dir="/nfs/ghome/live/martorellat/data/cifar_train",
        batch_size=TOTAL_SAMPLES,
        image_size=32,
        class_cond=True,
    )

	# dataloader = vl.build_streaming_image_dataloader(
	# 	datadir=f"/ceph/scratch/martorellat/data/{dataset_name}/mds/train",
	# 	batch_size=TOTAL_SAMPLES,
	# 	shuffle=False,
	# 	image_size=28 if dataset_name == "mnist" else 32,
	# 	num_channels=1 if dataset_name == "mnist" else 3,
	# 	transform=[transforms.Grayscale(1), transforms.ToTensor()]
	# 	if dataset_name == "mnist"
	# 	else [transforms.ToTensor()],
	# )

	# --- 2. Instantiate Correct Noise Schedule ---
	diffusion = create_gaussian_diffusion(
        learn_sigma=False,
        steps=4000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        
        # Distributional loss configuration
        use_distributional=False,
        distributional_track_terms_regardless_of_lambda=False, # can be set to False to save memory and compute, True will track all terms even if lambda is 0
        distributional_kernel="energy",
        distributional_lambda=1.0,
        distributional_lambda_weighting="constant",
        distributional_population_size=4,
        distributional_kernel_kwargs={"beta": 1.0},
        distributional_loss_weighting="no_weighting",  # can be "no_weighting" or "kingma_snr"
        distributional_num_eps_channels=3,
        dispersion_loss_type="none"
	)

	# --- 3. Prepare GUARANTEED DISJOINT Data & CACHE ---
	print(f"Fetching one large batch of {TOTAL_SAMPLES} samples to ensure disjoint sets...")
	try:
		full_batch, cond = next(iter(data))
		full_batch = full_batch.to(DEVICE)
	except StopIteration:
		raise ValueError(f"Dataloader could not provide enough samples. Need at least {TOTAL_SAMPLES}.")
	if full_batch.shape[0] < TOTAL_SAMPLES:
		raise ValueError(f"Dataloader only provided {full_batch.shape[0]} samples, but {TOTAL_SAMPLES} are required.")

	x0_importance_batch = full_batch[:M_IMPORTANCE]
	x0_benchmark_batch = full_batch[M_IMPORTANCE:TOTAL_SAMPLES]

	m_importance_samples = x0_importance_batch.view(M_IMPORTANCE, -1)
	x0_benchmark_samples = x0_benchmark_batch.view(N_BENCHMARK, -1)

	assert torch.sum(torch.all(m_importance_samples == x0_benchmark_samples[0], dim=1)) == 0, (
		"Data contamination detected! Benchmark and Importance sets are not disjoint."
	)
	print("Data sets successfully verified to be disjoint.")

	M, D = m_importance_samples.shape
	print(f"Data dimension D = {D}")
	x0_importance_norms_sq = torch.sum(m_importance_samples.pow(2), dim=1)

	# --- 4. Main Estimation Loop ---
	results_by_bin = {i: [] for i in range(NUM_T_BINS)}
	all_trace_variances = []
	all_norm_sq_variances = []

	print("\nStarting variance estimation loop...")
	for i in tqdm(range(N_BENCHMARK), desc="Estimating Irreducible Variance"):
		x0 = x0_benchmark_samples[i].unsqueeze(0)
		t_float = torch.randint(0, diffusion.num_timesteps-1, (1,)).item()
		t_tensor = torch.tensor([t_float], device=DEVICE)
		xt = diffusion.q_sample(x_start=x0, t=t_tensor)
		log_prob_fn = create_log_prob_function_optimized(diffusion, t_float, x0_importance_norms_sq, D)
		_, var_estimate = estimate_conditional_variance(
			x0_samples=m_importance_samples, xt=xt, p_xt_given_x0_log_prob=log_prob_fn
		)

		# Calculate BOTH quantities
		trace_var = torch.sum(var_estimate).item()
		norm_sq_var = torch.sum(var_estimate.pow(2)).item()

		# if USE_LOSS_WEIGHTING:
		# 	schedule_t = noise_schedule(t_tensor)
		# 	trace_var = loss_weighting_fn(trace_var, schedule_t).cpu()
		# 	norm_sq_var = loss_weighting_fn(norm_sq_var, schedule_t).cpu()

		all_trace_variances.append(trace_var)
		all_norm_sq_variances.append(norm_sq_var)
	
		t_float = t_float / diffusion.num_timesteps  # Normalize t to [0, 1]

		bin_idx = min(int(t_float * NUM_T_BINS), NUM_T_BINS - 1)
		results_by_bin[bin_idx].append((trace_var, norm_sq_var))

	# --- 5. Process and Aggregate Results ---
	V_irreducible_trace = np.mean(all_trace_variances)
	V_irreducible_norm_sq = np.mean(all_norm_sq_variances)

	bin_centers = (np.arange(NUM_T_BINS) + 0.5) / NUM_T_BINS
	avg_trace_in_bin = np.array(
		[np.mean([item[0] for item in results_by_bin[i]]) if results_by_bin[i] else 0 for i in range(NUM_T_BINS)]
	)
	std_trace_in_bin = np.array(
		[np.std([item[0] for item in results_by_bin[i]]) if results_by_bin[i] else 0 for i in range(NUM_T_BINS)]
	)
	avg_norm_sq_in_bin = np.array(
		[np.mean([item[1] for item in results_by_bin[i]]) if results_by_bin[i] else 0 for i in range(NUM_T_BINS)]
	)

	# --- 6. Visualize the Results ---
	print("\nGenerating plots...")
	plt.style.use("seaborn-v0_8-whitegrid")
	
	# fig, axs = plt.subplots(2, 2, figsize=(22, 8))
	fig, axs = plt.subplots(1, 1, figsize=(22 / 2, 5))
	axs_flat = [None, None, axs]
	# axs_flat = axs.flatten()
	# fig.suptitle(f"Empirical Total Loss Decomposition ({dataset_name})", fontsize=16)

	# # Plot 1: Trace of Variance (Directly comparable to MSE Loss)
	# axs_flat[0].plot(bin_centers, avg_trace_in_bin, marker="o", linestyle="-", label="Avg. Trace(Var) per Time Bin", zorder=2)
	# axs_flat[0].axhline(
	# 	y=V_irreducible_trace,
	# 	color="r",
	# 	linestyle="--",
	# 	label=f"Overall Avg. Trace(Var) = {V_irreducible_trace:.4f}",
	# 	zorder=1,
	# )
	# axs_flat[0].set_title("Irreducible Loss (Trace of Variance)", fontsize=16)
	# axs_flat[0].set_xlabel("Normalized Time (t)", fontsize=12)
	# axs_flat[0].set_ylabel("E[Tr(Cov(x₀|xₜ))]", fontsize=14)
	# axs_flat[0].legend(fontsize=10)

	# # Plot 2: Squared Norm of Variance
	# axs_flat[1].plot(
	# 	bin_centers,
	# 	avg_norm_sq_in_bin,
	# 	marker="o",
	# 	linestyle="-",
	# 	color="g",
	# 	label="Avg. ||Var||² per Time Bin",
	# 	zorder=2,
	# )
	# axs_flat[1].axhline(
	# 	y=V_irreducible_norm_sq,
	# 	color="m",
	# 	linestyle="--",
	# 	label=f"Overall Avg. ||Var||² = {V_irreducible_norm_sq:.4f}",
	# 	zorder=1,
	# )
	# axs_flat[1].set_title("Magnitude of Uncertainty (Squared Norm of Variance)", fontsize=16)
	# axs_flat[1].set_xlabel("Normalized Time (t)", fontsize=12)
	# axs_flat[1].set_ylabel("E[||Var(x₀|xₜ)||^2]", fontsize=14)
	# axs_flat[1].legend(fontsize=10)

	# Plot 3: Trace of Variance (Directly comparable to MSE Loss)
	axs_flat[2].plot(
		bin_centers,
		avg_trace_in_bin / D,
		marker="o",
		linestyle="-",
		label="Avg. E[Var(x₀|xₜ))] per Time Bin - normalized per pixel",
		zorder=2,
		alpha=0.8,
	)
	axs_flat[2].fill_between(
		bin_centers,
		(avg_trace_in_bin - std_trace_in_bin) / D,
		(avg_trace_in_bin + std_trace_in_bin) / D,
		alpha=0.2,
		color="tab:blue",
		label="Std. Dev.",
	)
	axs_flat[2].axhline(
		y=V_irreducible_trace / D,
		color="r",
		linestyle="--",
		label=f"Overall E[Var(x₀|xₜ))] = {V_irreducible_trace / D:.4f}",
		zorder=1,
	)
	axs_flat[2].set_title(f"Empirical Irreducible Variance ({dataset_name})", fontsize=16)
	axs_flat[2].set_xlabel("Normalized Time (t)", fontsize=12)
	axs_flat[2].set_ylabel("E[Var(x₀|xₜ))]", fontsize=14)
	axs_flat[2].legend(fontsize=10, framealpha=1, frameon=True)

	# # Plot 4: Squared Norm of Variance
	# axs_flat[3].plot(
	# 	bin_centers,
	# 	avg_norm_sq_in_bin / D,
	# 	marker="o",
	# 	linestyle="-",
	# 	color="g",
	# 	label="Avg. ||Var||² per Time Bin",
	# 	zorder=2,
	# )
	# axs_flat[3].axhline(
	# 	y=V_irreducible_norm_sq / D,
	# 	color="m",
	# 	linestyle="--",
	# 	label=f"Overall Avg. ||Var||² = {V_irreducible_norm_sq / D:.4f}",
	# 	zorder=1,
	# )
	# axs_flat[3].set_title("Magnitude of Uncertainty (Squared Norm of Variance)", fontsize=16)
	# axs_flat[3].set_xlabel("Normalized Time (t)", fontsize=12)
	# axs_flat[3].set_ylabel("E[||Var(x₀|xₜ)||^2]", fontsize=14)
	# axs_flat[3].legend(fontsize=10)

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(f"../outputs/variance_analysis_plot_{dataset_name}.png")
	plt.show()

	# --- 7. Print Final Results ---
	print("\n--- Final Results ---")
	print(f"Irreducible MSE Loss Component (Avg. Trace of Variance): {V_irreducible_trace:.4f}")
	print(
		" -> This is the theoretical best-case MSE your regression model can achieve. Compare your model's validation loss to this value."
	)

	print(f"\nMagnitude of Uncertainty (Avg. Squared Norm of Variance): {V_irreducible_norm_sq:.4f}")
	print(" -> This metric quantifies the overall 'spread' of the conditional distribution.")

#%%
if __name__ == "__main__":
	# To make this script runnable, we call main here.
	# In a real scenario, you would import and call this function as needed.
	main()

# %%
