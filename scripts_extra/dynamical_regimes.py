# %%
import itertools
import math
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from guided_diffusion.script_util import create_gaussian_diffusion
from scripts_extra.data_utils import load_data


# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================


def load_dataset(
	class_indices: list, num_samples_per_class: int, dataset_name: str, channel_to_use: Optional[int] = None
) -> torch.Tensor:
	"""
	Loads and preprocesses a subset of the CIFAR-10 dataset.

	Args:
		class_indices: A list of two integers for the classes to select (e.g., [3, 5] for cat, dog).
		num_samples_per_class: The number of samples to use from each class.

	Returns:
		A flattened and centered data tensor of shape (n_total_samples, d_features).
	"""
	print(f"Loading CIFAR-10 data for classes {class_indices}...")
	
	data = load_data(
		data_dir="/nfs/ghome/live/martorellat/data/cifar_train",
		batch_size=500,
		image_size=32,
		class_cond=True,
	)

	# Filter for the desired classes
	all_data = []
	all_labels = []
	
	assert len(data.dataset) % 500 == 0, "Batch size must be a multiple of 500 for CIFAR-10."
	
	iter_loader = iter(data)
	for _ in tqdm(range(100)):  # 50_000 / 500 (batch_size)
		img_batch, cond = next(iter_loader)
		labels = cond["y"]
		for i in range(labels.shape[0]):
			if labels[i].item() in class_indices:
				all_data.append(img_batch[i])
				all_labels.append(labels[i])

	all_data = torch.stack(all_data)
	all_labels = torch.stack(all_labels)

	# Flatten the images and select the specified number of samples
	n_total, c, h, w = all_data.shape
	# d_features = c * h * w
	d_features = c * h * w
	if channel_to_use is not None:
		d_features = 1 * h * w
		flat_data = all_data[:, channel_to_use, ...].view(n_total, d_features)
	else:
		flat_data = all_data.view(n_total, d_features)

	# Subsample the data
	subset_data = []
	for class_idx in class_indices:
		# Find indices for the current class
		class_mask = all_labels == class_idx
		class_flat_data = flat_data[class_mask]

		# Take the specified number of samples
		num_to_take = min(num_samples_per_class, len(class_flat_data))
		subset_data.append(class_flat_data[:num_to_take])
		print(f"  - Selected {num_to_take} samples for class {class_idx}.")

	final_data = torch.cat(subset_data, dim=0)

	# Center the data (subtract mean) for covariance calculation
	data_mean = torch.mean(final_data, dim=0, keepdim=True)
	centered_data = final_data - data_mean
	
	print(data_mean.min(), data_mean.max(), data_mean.mean(), data_mean.std())

	print(f"Dataset created with shape: {centered_data.shape}")
	return centered_data


# ==============================================================================
# 2. CALCULATION FUNCTIONS (PYTORCH / GPU)
# ==============================================================================


def calculate_speciation_time_torch(data: torch.Tensor) -> float:
	"""
	Calculates the speciation time (t_S) on the GPU.
	t_S = 0.5 * log(A), where A is the largest eigenvalue of the covariance matrix.
	"""
	print("\nCalculating speciation time (t_S)...")

	# torch.cov expects features in rows, so we transpose the data
	cov_matrix = torch.cov(data.T).to(torch.float64)

	# Use torch.linalg.eigvalsh for symmetric matrices (faster and more stable)
	eigenvalues = torch.linalg.eigvalsh(cov_matrix)

	# Get the largest eigenvalue (Lambda)
	lambda_max = torch.max(eigenvalues)

	if lambda_max <= 0:
		raise ValueError("The largest eigenvalue must be positive.")

	t_S = 0.5 * torch.log(lambda_max)

	print(f"  - Largest eigenvalue A = {lambda_max.item():.4f}")
	print(f"  - Calculated t_S = {t_S.item():.4f}")

	return t_S.item()


def _calculate_empirical_entropy_torch(
	data: torch.Tensor, t: float, delta_t: float, n_prime: int, batch_size: int
) -> float:
	"""
	Helper to estimate empirical entropy s^e(t) on GPU with batching.
	"""
	n_samples, d_features = data.shape
	device = data.device

	# 1. Generate n' noisy samples for Monte Carlo estimation
	rand_indices = torch.randint(0, n_samples, (n_prime,), device=device)
	a_mu = data[rand_indices, :]
	xi = torch.randn(n_prime, d_features, device=device)

	exp_minus_t = np.exp(-t)
	sqrt_delta_t = np.sqrt(delta_t)
	noisy_samples = a_mu * exp_minus_t + sqrt_delta_t * xi

	# 2. Prepare for batched calculation
	log_probabilities = torch.zeros(n_prime, device=device)
	log_pdf_const = -(d_features / 2.0) * np.log(2 * np.pi * delta_t)
	inv_2_delta_t = 1.0 / (2 * delta_t)
	all_means = data * exp_minus_t

	# 3. Process in batches
	num_batches = (n_prime + batch_size - 1) // batch_size
	for i in range(num_batches):
		start_idx = i * batch_size
		end_idx = min((i + 1) * batch_size, n_prime)
		noisy_batch = noisy_samples[start_idx:end_idx]

		# Highly efficient all-pairs squared distance calculation on GPU
		# ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2
		dists_sq = (
			torch.sum(noisy_batch**2, dim=1, keepdim=True)
			- 2 * (noisy_batch @ all_means.T)
			+ torch.sum(all_means**2, dim=1)
		)

		log_pdfs_for_batch = log_pdf_const - dists_sq * inv_2_delta_t
		logsumexp_vals = torch.logsumexp(log_pdfs_for_batch, dim=1)

		log_probabilities[start_idx:end_idx] = -torch.log(torch.tensor(n_samples, device=device)) + logsumexp_vals

	# 4. Final entropy calculation
	s_e = -torch.sum(log_probabilities) / (n_prime * d_features)
	return s_e.item()


def calculate_collapse_time_torch(data: torch.Tensor, t_range: np.ndarray, n_prime: int, batch_size: int) -> tuple:
	"""
	Calculates the collapse time (t_C) on the GPU.
	"""
	print(f"\nCalculating collapse time (t_C) over {len(t_range)} time steps...")
	n_samples, d_features = data.shape
	f_e_values = np.zeros_like(t_range, dtype=float)

	start_time = time.time()
	for i, t in enumerate(tqdm(t_range, desc="Collapse Time Calc")):
		if t <= 1e-6:
			f_e_values[i] = np.nan
			continue

		delta_t = 1 - np.exp(-2 * t)
		s_sep = (np.log(n_samples) / d_features) + 0.5 * (1 + np.log(2 * np.pi * delta_t))
		s_e = _calculate_empirical_entropy_torch(data, t, delta_t, n_prime, batch_size)
		f_e_values[i] = s_sep - s_e

	end_time = time.time()
	print(f"Calculation finished in {end_time - start_time:.2f} seconds.")

	# Find the zero-crossing point
	clean_indices = ~np.isnan(f_e_values)
	cross_indices = np.where(np.diff(np.sign(f_e_values[clean_indices])) < 0)[0]

	if len(cross_indices) == 0:
		print("Warning: No zero-crossing found for f^e(t). Try adjusting t_range.")
		return None, f_e_values

	original_indices = np.where(clean_indices)[0]
	idx1 = original_indices[cross_indices[-1]]
	idx2 = idx1 + 1

	t1, t2 = t_range[idx1], t_range[idx2]
	f1, f2 = f_e_values[idx1], f_e_values[idx2]
	t_C = t1 - f1 * (t2 - t1) / (f2 - f1)

	print(f"\n  - Zero-crossing found between t={t1:.3f} and t={t2:.3f}")
	print(f"  - Calculated t_C = {t_C:.4f}")

	return t_C, f_e_values


#%%
# # ==============================================================================
# # 3. MAIN EXECUTION BLOCK
# # ==============================================================================
# classes = [1, 7]  # Default classes: 3=cat, 5=dog
# num_samples = 30000
# dataset_name = "cifar10"

# # --- Setup Device ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # --- Load Data ---
# filtered_data = load_dataset(class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name).to(
# 	device
# )
# filtered_data_one_channel = load_dataset(
# 	class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name, channel_to_use=0
# ).to(device)
# n_samples_total, d_features = filtered_data.shape

# # %%
# # --- Run Calculations (Speciation) ---
# t_S_biroli_one_channel = calculate_speciation_time_torch(filtered_data_one_channel)
# t_S_biroli = calculate_speciation_time_torch(filtered_data)

# # %%
# # --- Run Calculations (Collapse) ---
# n_prime = 100000  # Number of Monte Carlo samples for entropy estimation
# batch_size = 1024 * 8

# # Scan a range of time points. Adjust based on t_S_biroli.
# t_scan_range = np.linspace(0.01, t_S_biroli + 1.0, 500)

# t_C_biroli, f_values = calculate_collapse_time_torch(filtered_data, t_scan_range, n_prime, batch_size)

# # %%

# # --- Plot the Results ---
# if t_C_biroli is not None:
# 	alpha = np.log(n_samples_total) / d_features
# 	f_values_normalized = f_values / alpha

# 	plt.figure(figsize=(10, 6))
# 	plt.plot(t_scan_range, f_values_normalized, "o-", label=r"$f^e(t) / \alpha$")
# 	plt.axvline(x=t_C_biroli, color="r", linestyle="--", label=f"Calculated $t_C = {t_C_biroli:.3f}$")
# 	plt.axvline(
# 		x=t_S_biroli_one_channel,
# 		color="k",
# 		linestyle="--",
# 		label=f"Calculated $t_S (1ch) = {t_S_biroli_one_channel:.3f}$",
# 	)
# 	plt.axvline(x=t_S_biroli, color="g", linestyle="--", label=f"Calculated $t_S = {t_S_biroli:.3f}$")
# 	plt.axhline(y=0, color="k", linestyle=":", linewidth=0.8)

# 	plt.title(f"Empirical Excess Entropy Density ({dataset_name})", fontsize=16)
# 	plt.xlabel("Time (t)", fontsize=12)
# 	plt.ylabel(r"Normalized Excess Entropy ($f^e(t) / \alpha$)", fontsize=12)
# 	plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# 	plt.legend()

# 	clean_f_vals = f_values_normalized[~np.isnan(f_values_normalized)]
# 	if len(clean_f_vals) > 0:
# 		plt.ylim(min(clean_f_vals) - 0.2, max(clean_f_vals) + 0.2)
# 	plt.xlim(0, t_scan_range[-1])

# 	plt.savefig("cifar10_collapse_plot.png")
# 	print("\nPlot saved to cifar10_collapse_plot.png")
# 	plt.show()

# %%


# Your provided CosineSchedule class
# I've modified it slightly to accept numpy arrays for plotting
from guided_diffusion.script_util import create_gaussian_diffusion
import torch
class CosineSchedule:
	def __init__(self):
		self.diffusion = create_gaussian_diffusion(
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
		
		self.logsnrs = torch.tensor(
			list(self._compute_lognsr(t) for t in range(4000))
		)

	def _compute_lognsr(self, t):
		# Allow both torch tensors and numpy arrays
		alpha = self.diffusion.sqrt_alphas_cumprod[t] 
		sigma = self.diffusion.sqrt_one_minus_alphas_cumprod[t]

		logsnr = torch.log(torch.tensor((alpha ** 2 / sigma ** 2)).clamp(min=1e-20))
		return logsnr


# The conversion function using the analytical formula we derived
def convert_biroli_to_custom_cosine(t_biroli: float, schedule: CosineSchedule, eps: float = 1e-20) -> float:
	"""
	Converts a time from the Biroli [0, inf) scale to your custom cosine schedule's [0, 1] scale.
	This function works by equating the Signal-to-Noise Ratio (SNR) between the two frameworks.

	Args:
		t_biroli: The time to convert (e.g., t_S or t_C) from the unnormalized [0, inf) scale.
		schedule: An instance of your CosineSchedule class, containing schedule parameters t_min and t_max.
		eps: A small epsilon to ensure numerical stability by preventing log(0) or division by zero.

	Returns:
		The equivalent time t_your on the normalized [0, 1] scale of your cosine schedule.
	"""
	# =======================================================================================================
	# STEP 1: Convert the Biroli time into its equivalent, schedule-independent log(SNR) value.
	# This is the universal "language" that connects the two time scales.
	#
	# DERIVATION:
	# 1. Biroli Forward Process: x(t) = x_0 * e^(-t) + sqrt(1 - e^(-2t)) * noise
	# 2. Signal Power (variance): (e^(-t))^2 = e^(-2t)
	# 3. Noise Power (variance): (sqrt(1 - e^(-2t)))^2 = 1 - e^(-2t)
	# 4. SNR_biroli(t) = Signal Power / Noise Power = e^(-2t) / (1 - e^(-2t))
	# 5. target_logSNR = log(SNR_biroli(t))
	# =======================================================================================================
	exp_m2t = np.exp(-2 * t_biroli)
	target_logSNR = np.log((exp_m2t + eps) / (1 - exp_m2t + eps))

	# =======================================================================================================
	# STEP 2: Find the time `t_your` in your schedule that produces this `target_logSNR`.
	# This requires analytically inverting your CosineSchedule's formula.
	#
	# DERIVATION:
	# 1. Your schedule's formula: logSNR = -2 * log(tan(t_min + t_your * (t_max - t_min)))
	# 2. Rearrange to solve for t_your:
	#    -0.5 * logSNR = log(tan( ... ))
	#    exp(-0.5 * logSNR) = tan( ... )
	#    arctan(exp(-0.5 * logSNR)) = t_min + t_your * (t_max - t_min)
	#    arctan(exp(-0.5 * logSNR)) - t_min = t_your * (t_max - t_min)
	#    t_your = (arctan(exp(-0.5 * logSNR)) - t_min) / (t_max - t_min)
	# =======================================================================================================

	# This line implements exp(-0.5 * target_logSNR)
	# exp_val = np.exp(-0.5 * target_logSNR)

	# # This line implements arctan(...)
	# atan_val = np.arctan(exp_val)

	# # This line implements the final division to isolate t_your
	# t_your = (atan_val - schedule.t_min) / (schedule.t_max - schedule.t_min)
	
	
	closest_discrete_timestep = np.argmin(np.abs(schedule.logsnrs - target_logSNR))
	t_your = closest_discrete_timestep / 4000

	# --- User feedback ---
	print(f"  - t_biroli = {t_biroli:.4f} -> target logSNR = {target_logSNR:.4f}")
	print(f"  - Calculated t_your = {t_your:.4f}")

	return t_your


# --- Main Demonstration ---

# 1. Instantiate your schedule
# These are the default values, you can change them if you use different ones.
# custom_schedule = CosineSchedule()

# # 2. Use computed Biroli times to convert them to your schedule
# print("Converting characteristic times for your Custom Cosine Schedule.")
# print("-" * 60)

# print("Converting Speciation Time (t_S):")
# t_S_your = convert_biroli_to_custom_cosine(t_S_biroli, custom_schedule)

# print("\n" + "-" * 60)
# print("Converting Speciation Time (t_S_one_channel):")
# t_S_your_one_channel = convert_biroli_to_custom_cosine(t_S_biroli_one_channel, custom_schedule)

# print("\n" + "-" * 60)
# print("Converting Collapse Time (t_C):")
# t_C_your = convert_biroli_to_custom_cosine(t_C_biroli, custom_schedule)

# print("\n" + "=" * 70)
# print("                       SUMMARY OF RESULTS")
# print("=" * 70)
# print(f"| {'Time':<12} | {'Biroli Scale [0, ∞)':<25} | {'Your Cosine Schedule [0, 1]':<30} |")
# print(f"|{'-' * 14}|{'-' * 27}|{'-' * 32}|")
# print(f"| {'Speciation':<12} | {t_S_biroli:<25.4f} | {t_S_your:<30.4f} |")
# print(f"| {'Speciation (1ch)':<12} | {t_S_biroli_one_channel:<25.4f} | {t_S_your_one_channel:<30.4f} |")
# print(f"| {'Collapse':<12} | {t_C_biroli:<25.4f} | {t_C_your:<30.4f} |")
# print("=" * 70)

# # Optional: Plot the schedule to visualize the conversion
# t_range_your = np.linspace(0.0, 1.0, 4000)
# logsnr_range = custom_schedule.logsnrs.numpy()

# logsnr_S = np.log(np.exp(-2 * t_S_biroli) / (1 - np.exp(-2 * t_S_biroli)))
# logsnr_S_one_channel = np.log(np.exp(-2 * t_S_biroli_one_channel) / (1 - np.exp(-2 * t_S_biroli_one_channel)))
# logsnr_C = np.log(np.exp(-2 * t_C_biroli) / (1 - np.exp(-2 * t_C_biroli)))

# plt.figure(figsize=(10, 6))
# plt.plot(t_range_your, logsnr_range, label="Your schedule: logSNR vs. t_your")
# plt.scatter(t_C_your, logsnr_C, color="red", zorder=5)
# plt.scatter(t_S_your_one_channel, logsnr_S_one_channel, color="black", zorder=5, label="t_S (1ch)")
# plt.scatter(t_S_your, logsnr_S, color="green", zorder=5)
# plt.axhline(
# 	y=logsnr_S_one_channel,
# 	color="black",
# 	linestyle="--",
# 	alpha=0.7,
# 	label=f"$t_S (1ch)$ Level (logSNR={logsnr_S_one_channel:.2f})",
# )
# plt.axhline(y=logsnr_S, color="green", linestyle="--", alpha=0.7, label=f"$t_S$ Level (logSNR={logsnr_S:.2f})")
# plt.axhline(y=logsnr_C, color="red", linestyle="--", alpha=0.7, label=f"$t_C$ Level (logSNR={logsnr_C:.2f})")

# plt.xlabel("Your Normalized Time t_your [0, 1]")
# plt.ylabel("log(SNR)")
# plt.title(f"Mapping Biroli Times to Your Cosine Noise Schedule - {dataset_name}")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# %%


# class KingmaSNRLossWeighting:
# 	"""Kingma et al. (2021) SNR loss weighting."""

# 	def __init__(self, b=0):
# 		self.b = b

# 	def __call__(self, loss, schedule_t):
# 		logsnr = schedule_t["log_snr"]
# 		weight = (1 + (self.b - logsnr).exp()) ** -1

# 		return loss * weight


# t_loss = torch.linspace(0, 1, 1000)
# schedule = CosineScheduleOriginal()
# schedule_t = schedule(t_loss)

# b_values = [-2, -1, 0, 1, 2]  # Different b values for Kingma SNR loss weighting
# loss_weighting = [KingmaSNRLossWeighting(b=b) for b in b_values]

# losses = torch.ones_like(t_loss)  # Example losses, can be any tensor
# weighted_losses = [lw(losses, schedule_t) for lw in loss_weighting]

# ==============================================================================
#                 STANDALONE PLOTTING SNIPPET
# ==============================================================================
# Assumes all required variables are pre-defined in your environment.

# 1. Create the figure and the primary (left) y-axis
# fig, ax1 = plt.subplots(figsize=(14, 8))

# # 2. Plot the Loss Weighting curves on the left Y-axis (ax1)
# color_loss = "tab:blue"
# ax1.set_xlabel("Your Normalized Time t_your [0, 1]", fontsize=14)
# ax1.set_ylabel("Loss Weight / Value", color=color_loss, fontsize=14)

# # Plot each weighted loss curve from your previous calculation
# for i, weighted_loss in enumerate(weighted_losses):
# 	ax1.plot(t_loss.numpy(), weighted_loss.numpy(), label=f"Weighted Loss (b={b_values[i]})")

# ax1.tick_params(axis="y", labelcolor=color_loss)
# ax1.set_ylim(-0.05, 1.05)
# ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# # 3. Create a secondary (right) y-axis that shares the same x-axis
# ax2 = ax1.twinx()

# # 4. Plot the log(SNR) curve on the right Y-axis (ax2)
# color_snr = "tab:red"
# ax2.set_ylabel("log(SNR)", color=color_snr, fontsize=14)
# (line_logsnr,) = ax2.plot(
# 	t_range_your, logsnr_range, color=color_snr, linestyle=":", linewidth=2.5, label="Schedule log(SNR)"
# )
# ax2.tick_params(axis="y", labelcolor=color_snr)
# ax2.set_ylim(schedule.logsnr_min - 1, schedule.logsnr_max + 1)

# # 5. Add vertical lines for the characteristic times t_S and t_C
# #    These lines will span both y-axes, connecting the concepts
# line_tc = plt.axvline(x=t_C_your, color="purple", linestyle="--", linewidth=2, label=f"$t_C = {t_C_your:.2f}$")
# line_ts1 = plt.axvline(
# 	x=t_S_your_one_channel,
# 	color="saddlebrown",
# 	linestyle="--",
# 	linewidth=2,
# 	label=f"$t_S$ (1ch) $= {t_S_your_one_channel:.2f}$",
# )
# line_ts_all = plt.axvline(
# 	x=t_S_your, color="darkgreen", linestyle="--", linewidth=2, label=f"$t_S$ (All ch) $= {t_S_your:.2f}$"
# )

# # 6. Finalize the plot with a title and a combined legend
# fig.suptitle(f"Kingma Loss Weighting vs. Schedule log(SNR) with Characteristic Times - {dataset_name}", fontsize=18)

# # Collect handles and labels from all plot elements for a single, unified legend
# handles1, labels1 = ax1.get_legend_handles_labels()
# all_handles = handles1 + [line_logsnr, line_tc, line_ts1, line_ts_all]
# all_labels = labels1 + [line_logsnr.get_label(), line_tc.get_label(), line_ts1.get_label(), line_ts_all.get_label()]
# ax1.legend(all_handles, all_labels, loc="upper right")

# # Ensure the layout is clean and titles/labels don't overlap
# fig.tight_layout(rect=[0, 0, 1, 0.96])

# # 7. Display the plot
# plt.show()
# %%


def calculate_dynamical_regime_points(dataset_name, classes, num_samples=50000, n_prime=100000, batch_size=1024 * 8):
	# --- Setup Device ---
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	# --- Load Data ---
	filtered_data = load_dataset(
		class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name
	).to(device)
	filtered_data_one_channel = load_dataset(
		class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name, channel_to_use=0
	).to(device)
	n_samples_total, d_features = filtered_data.shape

	# --- Run Calculations (Speciation) ---
	t_S_biroli_one_channel = calculate_speciation_time_torch(filtered_data_one_channel)
	t_S_biroli = calculate_speciation_time_torch(filtered_data)

	# --- Run Calculations (Collapse) ---
	# Scan a range of time points. Adjust based on t_S_biroli.
	t_scan_range = np.linspace(0.01, t_S_biroli + 1.0, 500)

	t_C_biroli, f_values = calculate_collapse_time_torch(filtered_data, t_scan_range, n_prime, batch_size)

	return (t_C_biroli, t_S_biroli, t_S_biroli_one_channel, f_values, n_samples_total)


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
classes_pairs = list(itertools.combinations(classes, 2))
results = [
	calculate_dynamical_regime_points("cifar10", list(pair), num_samples=50000, n_prime=50000) for pair in classes_pairs
]

#%%
# save to pickle file
import pickle

dataset_name = "cifar10"
d_features = 32 * 32 * 3  # For CIFAR-10, each image is 32x32 pixels with 3 color channels
with open(f"../outputs/collapse_results_cifar10.pkl", "wb") as f:
	pickle.dump(results, f)
# %%


# for i, (t_C, t_S, t_S_one_channel, f_values, n_samples_total) in enumerate(results):
# 	alpha = np.log(n_samples_total) / d_features
# 	f_values_normalized = f_values / alpha

# 	plt.figure(figsize=(10, 6))
# 	plt.plot(t_scan_range, f_values_normalized, "o-", label=r"$f^e(t) / \alpha$")
# 	plt.axvline(x=t_C, color="r", linestyle="--", label=f"Calculated $t_C = {t_C:.3f}$")
# 	plt.axvline(
# 		x=t_S_one_channel,
# 		color="k",
# 		linestyle="--",
# 		label=f"Calculated $t_S (1ch) = {t_S_one_channel:.3f}$",
# 	)
# 	plt.axvline(x=t_S, color="g", linestyle="--", label=f"Calculated $t_S = {t_S:.3f}$")
# 	plt.axhline(y=0, color="k", linestyle=":", linewidth=0.8)

# 	plt.title(f"Empirical Excess Entropy Density ({dataset_name})", fontsize=16)
# 	plt.xlabel("Time (t)", fontsize=12)
# 	plt.ylabel(r"Normalized Excess Entropy ($f^e(t) / \alpha$)", fontsize=12)
# 	plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# 	plt.legend()

# 	clean_f_vals = f_values_normalized[~np.isnan(f_values_normalized)]
# 	if len(clean_f_vals) > 0:
# 		plt.ylim(min(clean_f_vals) - 0.2, max(clean_f_vals) + 0.2)
# 	plt.xlim(0, t_scan_range[-1])

# 	plt.savefig("cifar10_collapse_plot.png")
# 	print("\nPlot saved to cifar10_collapse_plot.png")
# 	plt.show()

# %%

n_samples_total = [x[4] for x in results]
t_C = np.average([result[0] for result in results], weights=n_samples_total)
t_S = np.average([result[1] for result in results], weights=n_samples_total)
t_S_one_channel = np.average([result[2] for result in results], weights=n_samples_total)
f_values = np.average([result[3] for result in results], axis=0, weights=n_samples_total)

t_scan_range = np.linspace(0.01, t_S + 1.0, 500)

alpha = np.log(np.average(n_samples_total)) / d_features
f_values_normalized = f_values / alpha

plt.figure(figsize=(10, 6))
plt.plot(t_scan_range, f_values_normalized, "o-", label=r"$f^e(t) / \alpha$")
plt.axvline(x=t_C, color="r", linestyle="--", label=f"Calculated $t_C = {t_C:.3f}$")
plt.axvline(
	x=t_S_one_channel,
	color="k",
	linestyle="--",
	label=f"Calculated $t_S (1ch) = {t_S_one_channel:.3f}$",
)
plt.axvline(x=t_S, color="g", linestyle="--", label=f"Calculated $t_S = {t_S:.3f}$")
plt.axhline(y=0, color="k", linestyle=":", linewidth=0.8)

plt.title(f"Empirical Excess Entropy Density ({dataset_name})", fontsize=16)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel(r"Normalized Excess Entropy ($f^e(t) / \alpha$)", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

clean_f_vals = f_values_normalized[~np.isnan(f_values_normalized)]
if len(clean_f_vals) > 0:
	plt.ylim(min(clean_f_vals) - 0.2, max(clean_f_vals) + 0.2)
plt.xlim(0, t_scan_range[-1])

plt.savefig("../outputs/cifar10_collapse_plot.png")
print("\nPlot saved to cifar10_collapse_plot.png")
plt.show()

# --- Main Demonstration ---

# 1. Instantiate your schedule
# These are the default values, you can change them if you use different ones.
custom_schedule = CosineSchedule()

# 2. Use computed Biroli times to convert them to your schedule
print("Converting characteristic times for your Custom Cosine Schedule.")
print("-" * 60)

print("Converting Speciation Time (t_S):")
t_S_your = convert_biroli_to_custom_cosine(t_S, custom_schedule)

print("\n" + "-" * 60)
print("Converting Speciation Time (t_S_one_channel):")
t_S_your_one_channel = convert_biroli_to_custom_cosine(t_S_one_channel, custom_schedule)

print("\n" + "-" * 60)
print("Converting Collapse Time (t_C):")
t_C_your = convert_biroli_to_custom_cosine(t_C, custom_schedule)

print("\n" + "=" * 70)
print("                       SUMMARY OF RESULTS")
print("=" * 70)
print(f"| {'Time':<12} | {'Biroli Scale [0, ∞)':<25} | {'Your Cosine Schedule [0, 1]':<30} |")
print(f"|{'-' * 14}|{'-' * 27}|{'-' * 32}|")
print(f"| {'Speciation':<12} | {t_S:<25.4f} | {t_S_your:<30.4f} |")
print(f"| {'Speciation (1ch)':<12} | {t_S_one_channel:<25.4f} | {t_S_your_one_channel:<30.4f} |")
print(f"| {'Collapse':<12} | {t_C:<25.4f} | {t_C_your:<30.4f} |")
print("=" * 70)

# Optional: Plot the schedule to visualize the conversion
t_range_your = np.linspace(0.0, 1.0, 4000)
logsnr_range = custom_schedule.logsnrs.numpy()

logsnr_S = np.log(np.exp(-2 * t_S) / (1 - np.exp(-2 * t_S)))
logsnr_S_one_channel = np.log(np.exp(-2 * t_S_one_channel) / (1 - np.exp(-2 * t_S_one_channel)))
logsnr_C = np.log(np.exp(-2 * t_C) / (1 - np.exp(-2 * t_C)))

plt.figure(figsize=(10, 6))
plt.plot(t_range_your, logsnr_range, label="Your schedule: logSNR vs. t_your")
plt.scatter(t_C_your, logsnr_C, color="red", zorder=5)
plt.scatter(t_S_your_one_channel, logsnr_S_one_channel, color="black", zorder=5, label="t_S (1ch)")
plt.scatter(t_S_your, logsnr_S, color="green", zorder=5)
plt.axhline(
	y=logsnr_S_one_channel,
	color="black",
	linestyle="--",
	alpha=0.7,
	label=f"$t_S (1ch)$ Level (logSNR={logsnr_S_one_channel:.2f})",
)
plt.axhline(y=logsnr_S, color="green", linestyle="--", alpha=0.7, label=f"$t_S$ Level (logSNR={logsnr_S:.2f})")
plt.axhline(y=logsnr_C, color="red", linestyle="--", alpha=0.7, label=f"$t_C$ Level (logSNR={logsnr_C:.2f})")

plt.xlabel("Your Normalized Time t_your [0, 1]")
plt.ylabel("log(SNR)")
plt.title(f"Mapping Biroli Times to Your Cosine Noise Schedule - {dataset_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"../outputs/{dataset_name}_time_matching_dynamical.png")
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
#                 STANDALONE PLOTTING SNIPPET WITH ERROR BARS
# ==============================================================================
# Assumes 'results', 'd_features', 't_scan_range', and 'dataset_name' are pre-defined.

# --- 1. Calculate Averages and Standard Deviations ---

# Extract raw values from each run
all_f_values = np.array([result[3] for result in results])
all_t_C = np.array([result[0] for result in results])
all_t_S = np.array([result[1] for result in results])
all_t_S_one_channel = np.array([result[2] for result in results])
n_samples_total = [x[4] for x in results]

# --- Calculate weighted averages (means) ---
f_values_mean = np.average(all_f_values, axis=0, weights=n_samples_total)
t_C_mean = np.average(all_t_C, weights=n_samples_total)
t_S_mean = np.average(all_t_S, weights=n_samples_total)
t_S_one_channel_mean = np.average(all_t_S_one_channel, weights=n_samples_total)

# --- Calculate standard deviations ---
# For plotting, a simple std is sufficient and visually clear.
f_values_std = np.std(all_f_values, axis=0)
t_C_std = np.std(all_t_C)
t_S_std = np.std(all_t_S)
t_S_one_channel_std = np.std(all_t_S_one_channel)

# Normalize the entropy values
alpha = np.log(np.average(n_samples_total)) / d_features
f_values_normalized_mean = f_values_mean / alpha
f_values_normalized_std = f_values_std / alpha

# --- 2. Create the Plot ---

plt.figure(figsize=(12, 7))

# --- Plot the main curve and its error band ---
# Main line for the average
plt.plot(t_scan_range, f_values_normalized_mean, "o-", label=r"Mean $f^e(t) / \alpha$", zorder=5)

# Shaded region for the standard deviation
plt.fill_between(
	t_scan_range,
	f_values_normalized_mean - f_values_normalized_std,
	f_values_normalized_mean + f_values_normalized_std,
	alpha=0.2,
	color="tab:blue",
	label=r"Std. Dev. of $f^e(t)$",
)

# --- Plot the vertical lines for characteristic times and their error bands ---
# t_C
plt.axvline(x=t_C_mean, color="r", linestyle="--", label=f"Mean $t_C = {t_C_mean:.3f}$", zorder=5)
plt.axvspan(t_C_mean - t_C_std, t_C_mean + t_C_std, alpha=0.2, color="red", label=r"Std. Dev. of $t_C$")

# t_S (one channel)
plt.axvline(
	x=t_S_one_channel_mean, color="k", linestyle="--", label=f"Mean $t_S (1ch) = {t_S_one_channel_mean:.3f}$", zorder=5
)
plt.axvspan(
	t_S_one_channel_mean - t_S_one_channel_std,
	t_S_one_channel_mean + t_S_one_channel_std,
	alpha=0.2,
	color="gray",
	label=r"Std. Dev. of $t_S (1ch)$",
)

# t_S (all channels)
plt.axvline(x=t_S_mean, color="g", linestyle="--", label=f"Mean $t_S = {t_S_mean:.3f}$", zorder=5)
plt.axvspan(t_S_mean - t_S_std, t_S_mean + t_S_std, alpha=0.2, color="green", label=r"Std. Dev. of $t_S$")

# --- Final plot formatting ---
plt.axhline(y=0, color="k", linestyle=":", linewidth=0.8)
plt.title(f"Empirical Excess Entropy Density ({dataset_name})", fontsize=16)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel(r"Normalized Excess Entropy ($f^e(t) / \alpha$)", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Create a more organized legend
plt.legend(loc="best", fontsize="small")

# Set plot limits
clean_f_vals = f_values_normalized_mean[~np.isnan(f_values_normalized_mean)]
if len(clean_f_vals) > 0:
	plt.ylim(min(clean_f_vals) - 0.3, max(clean_f_vals) + 0.3)
plt.xlim(0, t_scan_range[-1])

# Save and show
plt.savefig(f"../outputs/{dataset_name}_collapse_plot_with_errors.png")
print(f"\nPlot saved to {dataset_name}_collapse_plot_with_errors.png")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#      STANDALONE PLOTTING SNIPPET FOR LOG(SNR) CURVE WITH ERROR BARS
# ==============================================================================
# Assumes the mean and std dev for t_C, t_S, and t_S_one_channel are pre-defined.

# --- 1. Propagate the error from the Biroli scale to your custom scale ---

def get_error_bounds(t_mean, t_std, schedule):
	"""
	Calculates the mean point and its error bounds on the custom schedule scale.
	"""
	# Define the range of the Biroli time based on its standard deviation
	t_biroli_min = t_mean - t_std
	t_biroli_max = t_mean + t_std

	# Convert the mean and the bounds to your custom [0, 1] time scale
	t_your_mean = convert_biroli_to_custom_cosine(t_mean, schedule)
	t_your_min = convert_biroli_to_custom_cosine(t_biroli_min, schedule)
	t_your_max = convert_biroli_to_custom_cosine(t_biroli_max, schedule)

	# Calculate the corresponding logSNR values
	# Note the inverse relationship: a smaller t_biroli gives a larger logSNR
	logsnr_mean = np.log(np.exp(-2 * t_mean) / (1 - np.exp(-2 * t_mean)))
	logsnr_min = np.log(np.exp(-2 * t_biroli_max) / (1 - np.exp(-2 * t_biroli_max))) # max t -> min logSNR
	logsnr_max = np.log(np.exp(-2 * t_biroli_min) / (1 - np.exp(-2 * t_biroli_min))) # min t -> max logSNR

	# Create error values for plt.errorbar (which expects [lower_err, upper_err])
	x_err = [[t_your_mean - t_your_min], [t_your_max - t_your_mean]]
	y_err = [[logsnr_mean - logsnr_min], [logsnr_max - logsnr_mean]]
	
	return t_your_mean, logsnr_mean, x_err, y_err, logsnr_min, logsnr_max


# Calculate error bounds for each characteristic time
(t_C_your_mean, logsnr_C_mean, t_C_xerr, t_C_yerr, 
 logsnr_C_min, logsnr_C_max) = get_error_bounds(t_C_mean, t_C_std, custom_schedule)

(t_S_your_mean, logsnr_S_mean, t_S_xerr, t_S_yerr,
 logsnr_S_min, logsnr_S_max) = get_error_bounds(t_S_mean, t_S_std, custom_schedule)

(t_S_1ch_your_mean, logsnr_S_1ch_mean, t_S_1ch_xerr, t_S_1ch_yerr,
 logsnr_S_1ch_min, logsnr_S_1ch_max) = get_error_bounds(t_S_one_channel_mean, t_S_one_channel_std, custom_schedule)


# --- 2. Create the Plot ---

plt.figure(figsize=(12, 7))
t_range_your = np.linspace(0.0, 1.0, 4000)
logsnr_range = custom_schedule.logsnrs.numpy()

# Plot the main schedule curve
plt.plot(t_range_your, logsnr_range, label="Cosine schedule: logSNR vs. t", zorder=1)

# --- Plot the characteristic points with their error bars ---

# For t_C
plt.errorbar(t_C_your_mean, logsnr_C_mean, yerr=t_C_yerr, xerr=t_C_xerr,
			 fmt='o', color="red", capsize=5, label=f"Mean $t_C$", zorder=5)
plt.axhspan(logsnr_C_min, logsnr_C_max, color='red', alpha=0.15, label="Std. Dev. of $t_C$ Level")

# For t_S (one channel)
plt.errorbar(t_S_1ch_your_mean, logsnr_S_1ch_mean, yerr=t_S_1ch_yerr, xerr=t_S_1ch_xerr,
			 fmt='o', color="black", capsize=5, label=f"Mean $t_S (1ch)$", zorder=5)
plt.axhspan(logsnr_S_1ch_min, logsnr_S_1ch_max, color='gray', alpha=0.15, label="Std. Dev. of $t_S (1ch)$ Level")

# For t_S (all channels)
plt.errorbar(t_S_your_mean, logsnr_S_mean, yerr=t_S_yerr, xerr=t_S_xerr,
			 fmt='o', color="green", capsize=5, label=f"Mean $t_S$", zorder=5)
plt.axhspan(logsnr_S_min, logsnr_S_max, color='green', alpha=0.15, label="Std. Dev. of $t_S$ Level")

# --- 3. Final plot formatting ---
plt.xlabel("Normalized Time t [0, 1]", fontsize=12)
plt.ylabel("log(SNR)", fontsize=12)
plt.title(f"Mapping Biroli Times to Cosine Noise Schedule ({dataset_name})", fontsize=16)
plt.legend(loc='best', fontsize='small')
plt.grid(True, alpha=0.3)
plt.savefig(f"../outputs/{dataset_name}_time_matching_dynamical_with_error_bars.png")
plt.show()
# %%

# save to pickle file
import pickle
with open(f"../outputs/collapse_results_{dataset_name}.pkl", "wb") as f:
	pickle.dump(results, f)
# %%
import pickle
# load from pickle file
dataset_name = "cifar10"
with open(f"../outputs/collapse_results_{dataset_name}.pkl", "rb") as f:
	results = pickle.load(f)

# %%
import numpy as np
results
# %%
all_f_values = np.array([result[3] for result in results])
all_t_C = np.array([result[0] for result in results])
all_t_S = np.array([result[1] for result in results])
all_t_S_one_channel = np.array([result[2] for result in results])
n_samples_total = [x[4] for x in results]

schedule = CosineSchedule()
# %%
convert_biroli_to_custom_cosine(all_t_C.mean(), schedule), convert_biroli_to_custom_cosine(all_t_S.mean(), schedule), convert_biroli_to_custom_cosine(all_t_S_one_channel.mean(), schedule)
# %%
