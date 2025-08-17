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

from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import blobfile as bf

class ImageDataset(Dataset):
	def __init__(
		self,
		resolution,
		image_paths,
		classes=None,
		shard=0,
		num_shards=1,
		random_crop=False,
		random_flip=True,
	):
		super().__init__()
		self.resolution = resolution
		self.local_images = image_paths[shard:][::num_shards]
		self.local_classes = None if classes is None else classes[shard:][::num_shards]
		self.random_crop = random_crop
		self.random_flip = random_flip

	def __len__(self):
		return len(self.local_images)

	def __getitem__(self, idx):
		path = self.local_images[idx]
		with bf.BlobFile(path, "rb") as f:
			pil_image = Image.open(f)
			pil_image.load()
		pil_image = pil_image.convert("RGB")
		
		arr = transforms.ToTensor()(pil_image)

		out_dict = {}
		if self.local_classes is not None:
			out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
		return arr, out_dict

def load_data(
	*,
	data_dir,
	batch_size,
	image_size,
	class_cond=False,
):
	"""
	For a dataset, create a generator over (images, kwargs) pairs.

	Each images is an NCHW float tensor, and the kwargs dict contains zero or
	more keys, each of which map to a batched Tensor of their own.
	The kwargs dict can be used for class labels, in which case the key is "y"
	and the values are integer tensors of class labels.

	:param data_dir: a dataset directory.
	:param batch_size: the batch size of each returned pair.
	:param image_size: the size to which images are resized.
	:param class_cond: if True, include a "y" key in returned dicts for class
					   label. If classes are not available and this is true, an
					   exception will be raised.
	:param deterministic: if True, yield results in a deterministic order.
	:param random_crop: if True, randomly crop the images for augmentation.
	:param random_flip: if True, randomly flip the images for augmentation.
	"""
	if not data_dir:
		raise ValueError("unspecified data directory")
	all_files = _list_image_files_recursively(data_dir)
	classes = None
	if class_cond:
		# Assume classes are the first part of the filename,
		# before an underscore.
		class_names = [bf.basename(path).split("_")[0] for path in all_files]
		sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
		classes = [sorted_classes[x] for x in class_names]
	dataset = ImageDataset(
		image_size,
		all_files,
		classes=classes,
		random_crop=False,
		random_flip=False,
	)
	return DataLoader(
		dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
	)
	
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


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
				img = img_batch[i]
				img_flipped = transforms.Compose([
					transforms.ToPILImage(),
					transforms.RandomHorizontalFlip(1.0),
					transforms.ToTensor(),
				])(img)
				all_data.append(img)
				# all_data.append(img_flipped)
				all_labels.append(labels[i])
				# all_labels.append(labels[i])

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
	device = "cuda"
	
	# print(device)

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
	all_means = (data * exp_minus_t).to(device)

	# 3. Process in batches
	num_batches = (n_prime + batch_size - 1) // batch_size
	for i in range(num_batches):
		start_idx = i * batch_size
		end_idx = min((i + 1) * batch_size, n_prime)
		noisy_batch = noisy_samples[start_idx:end_idx].to(device)

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
	data = data.to("cuda")

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
	
# def calculate_separation_time_torch(data: torch.Tensor, batch_size: int = 256) -> float:
#     """
#     Calculates the "guaranteed separation time" (t_sep) on the GPU.
#     This is a stricter, geometry-based alternative to the entropy-based t_c.

#     Args:
#         data (torch.Tensor): The input dataset, shape (n_samples, d_features).
#         batch_size (int): The batch size to use for the pairwise distance calculation to manage memory.

#     Returns:
#         float: The calculated separation time t_sep.
#     """
#     print("\nCalculating Guaranteed Separation Time (t_sep)...")
#     n_samples = data.shape[0]
#     device = data.device
    
#     # --- Step 1: Find the minimum distance between any two points (d_min) ---
#     min_distances = torch.full((n_samples,), float('inf'), device=device)
    
#     start_time = time.time()
    
#     # Process in batches to avoid a massive n_samples x n_samples distance matrix
#     num_batches = (n_samples + batch_size - 1) // batch_size
#     for i in tqdm(range(num_batches), desc="Finding d_min"):
#         start_idx = i * batch_size
#         end_idx = min(start_idx + batch_size, n_samples)
#         batch = data[start_idx:end_idx]
        
#         # Calculate pairwise L2 distances between the current batch and the entire dataset
#         dists = torch.cdist(batch, data, p=2)
        
#         # Set diagonal to infinity so a point's distance to itself is not considered
#         # We need to be careful with indexing here for batches.
#         for j in range(batch.shape[0]):
#             dists[j, start_idx + j] = float('inf')

#         # Find the minimum distance for each point in the batch
#         min_dists_for_batch, _ = torch.min(dists, dim=1)
#         min_distances[start_idx:end_idx] = min_dists_for_batch

#     # The final d_min is the minimum of these minimum distances
#     d_min = torch.min(min_distances)
    
#     end_time = time.time()
#     print(f"  - Minimum distance calculation finished in {end_time - start_time:.2f} seconds.")
#     print(f"  - d_min (closest distance between any two points) = {d_min.item():.4f}")

#     # --- Step 2: Apply the analytical formula to solve for t_sep ---
#     # t_sep = -0.5 * log(1 - (d_min / 6)^2)
    
#     d_min_over_6_sq = (d_min / 6.0)**2
#     if d_min_over_6_sq >= 1.0:
#         print("Warning: d_min is too large, resulting in a non-real t_sep. The data points are extremely well-separated.")
#         return float('nan')

#     t_sep = -0.5 * torch.log(1 - d_min_over_6_sq)
    
#     print(f"  - Calculated t_sep = {t_sep.item():.4f}")
    
#     return t_sep.item()

def _find_d_min_torch(data: torch.Tensor, batch_size: int = 256, return_all_distances: bool = False):
    """
    Finds the minimum distance between any two distinct points in the dataset (d_min).
    This is a shared helper function.

    Args:
        data (torch.Tensor): The input dataset, shape (n_samples, d_features).
        batch_size (int): Batch size for memory-efficient distance calculation.
        return_all_distances (bool): If True, returns (d_min, all_distances). If False, returns just d_min.

    Returns:
        float or tuple: If return_all_distances=False, returns the minimum distance d_min.
                       If return_all_distances=True, returns (d_min, all_distances) where
                       all_distances is a tensor of all minimum distances for each point.
    """
    n_samples = data.shape[0]
    data = data.to("cuda")  # Ensure data is on GPU
    device = data.device
    min_distances = torch.full((n_samples,), float('-inf'), device=device)
    
    print(f"\nCalculating d_min (minimum distance between any two points)...")
    if return_all_distances:
        print("  - Will return all distances for quantile calculation")
    start_time = time.time()
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Finding d_min"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data[start_idx:end_idx].to("cuda")
        
        dists = torch.cdist(batch, data, p=2)
        
        # Set diagonal to infinity to ignore a point's distance to itself
        for j in range(batch.shape[0]):
            dists[j, start_idx + j] = float('inf')

        min_dists_for_batch, _ = torch.min(dists, dim=1)
        min_distances[start_idx:end_idx] = min_dists_for_batch

    d_min = torch.min(min_distances)
    
    end_time = time.time()
    print(f"  - d_min calculation finished in {end_time - start_time:.2f} seconds.")
    print(f"  - d_min = {d_min.item():.4f}")
    
    if return_all_distances:
        return d_min.item(), min_distances
    else:
        return d_min.item()

# ==============================================================================
# 2. UPDATED CALCULATION FUNCTIONS
# ==============================================================================

def calculate_separation_time_torch(data: torch.Tensor, batch_size: int = 256) -> float:
    """
    Calculates the "guaranteed separation time" (t_sep) on the GPU.
    Uses the shared _find_d_min_torch helper function.
    """
    print("\nCalculating Guaranteed Separation Time (t_sep)...")
    
    # --- Step 1: Find d_min using the shared helper ---
    d_min, all_distances = _find_d_min_torch(data, batch_size, return_all_distances=True)
	
    # print quantiles of all_distances
    print(f"  - All distances (quantiles): {torch.quantile(all_distances, torch.tensor([0.25, 0.5, 0.75]).to('cuda'), dim=0)}")

    # --- Step 2: Apply the analytical formula to solve for t_sep ---
    d_min_over_6_sq = (d_min / 6.0)**2
    if d_min_over_6_sq >= 1.0:
        print("Warning: d_min is too large, resulting in a non-real t_sep.")
        return float('nan')

    t_sep = -0.5 * np.log(1 - d_min_over_6_sq)
    
    print(f"  - Calculated t_sep = {t_sep:.4f}")
    
    return t_sep, d_min, all_distances

def ambiguity_ratio(t_biroli: np.ndarray, d: float, eps: float = 1e-20) -> np.ndarray:
    """
    Calculates the Ambiguity Ratio A(t) for a given unnormalized time t and distance d.

    The Ambiguity Ratio quantifies the influence of a neighboring data point, at distance d,
    on a trajectory collapsing towards its target. It is defined as:
    A(t) = exp(-d^2 / (2 * (exp(2t) - 1)))

    Args:
        t_biroli (np.ndarray): An array of unnormalized time points t on the [0, inf) scale.
        d (float): The distance between the target data point and its nearest neighbor.
        eps (float): A small epsilon for numerical stability.

    Returns:
        np.ndarray: The Ambiguity Ratio A(t) for each time point.
    """
    # Calculate the denominator term: 2 * (exp(2t) - 1)
    denominator = 2 * (np.exp(2 * t_biroli) - 1 + eps)
    
    # Calculate the exponent term: -d^2 / denominator
    exponent = -(d**2) / denominator
    
    # Return the final ratio
    return np.exp(exponent)

#%%
# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Default classes: 3=cat, 5=dog
num_samples = 300000
dataset_name = "cifar10"

# --- Setup Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Data ---
filtered_data = load_dataset(class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name)
# filtered_data_one_channel = load_dataset(
# 	class_indices=classes, num_samples_per_class=num_samples, dataset_name=dataset_name, channel_to_use=0
# ).to(device)
n_samples_total, d_features = filtered_data.shape

# %%
# --- Run Calculations (Speciation) ---
# t_S_biroli_one_channel = calculate_speciation_time_torch(filtered_data_one_channel)
t_S_biroli = calculate_speciation_time_torch(filtered_data)

#%%
t_C_strict, d_min, all_distances = calculate_separation_time_torch(filtered_data)

#%%
# Plotting all_distances against quantiles
plt.figure(figsize=(10, 6))
sorted_distances = torch.sort(all_distances)[0].cpu().numpy()
quantiles = np.linspace(0, 1, len(sorted_distances))
plt.plot(quantiles, sorted_distances)
plt.xlabel("Quantile")
plt.ylabel("Nearest Neighbor Distance")
plt.title("Distribution of Nearest Neighbor Distances")
plt.grid(True)
plt.show()


#%%


# ==============================================================================
# 2. NUMERICAL VERIFICATION
# ==============================================================================

# Create a range of unnormalized time points to test, from high t to low t
t_biroli_range = np.linspace(3, 0.01, 500)

# Calculate the Ambiguity Ratio over this time range
ambiguity_schedule = ambiguity_ratio(t_biroli_range, d=d_min)

# --- Plot the results to verify the properties ---
plt.figure(figsize=(10, 6))
plt.plot(t_biroli_range, ambiguity_schedule)

# Let's add some markers for where t_c and t_sep might be
t_C = 0.5
stds = 2.5
t_C_strict = -0.5 * np.log(1 - (d_min / (2 * stds))**2) # From our formula
t_C_stricter = -0.5 * np.log(1 - (d_min / (2 * 3))**2) # From our formula

plt.axvline(x=t_C, color='r', linestyle='--', label=f'Conceptual $t_C = {t_C:.3f}$')
plt.axvline(x=t_C_strict, color='g', linestyle='--', label=r'Conceptual $t_{sep} \text{ strict}' + f' = {t_C_strict:.3f}$')
plt.axvline(x=t_C_stricter, color='orange', linestyle='--', label=r'Conceptual $t_{sep} \text{ stricter}' + f' = {t_C_stricter:.3f}$')

plt.xlabel("Unnormalized Time t (Biroli Scale)")
plt.ylabel("Ambiguity Ratio A(t)")
plt.title(f"Numerical Verification of Ambiguity Ratio for d_min = {d_min}")
plt.grid(True)
plt.legend()
# Set x-axis to reverse to show the "forward" evolution in the reverse process
# plt.xlim(max(t_biroli_range), min(t_biroli_range)) 
plt.show()

# --- Print some key values to check the behavior ---
print("--- Verifying Properties ---")
print(f"Ambiguity at high t (t=3.0): A(t) = {ambiguity_ratio(np.array([3.0]), d=d_min)[0]:.4f} (should be close to 1)")
print(f"Ambiguity at t_C (t={t_C:.3f}): A(t) = {ambiguity_ratio(np.array([t_C]), d=d_min)[0]:.4f}")
print(f"Ambiguity at t_C strict (t={t_C_strict:.3f}): A(t) = {ambiguity_ratio(np.array([t_C_strict]), d=d_min)[0]:.4f}")
print(f"Ambiguity at low t (t=0.01): A(t) = {ambiguity_ratio(np.array([0.01]), d=d_min)[0]:.4f} (should be close to 0)")

# %%
#%%
# ==============================================================================
# COMPREHENSIVE AMBIGUITY RATIO HEATMAP VISUALIZATION
# ==============================================================================

# # Define the parameter space for the heatmap
# t_biroli_range = np.linspace(0.01, 2.5, 100)  # Time range
# d_range = np.linspace(d_min * 0.5, d_min * 3.0, 100)  # Distance range around d_min

# # Create meshgrid for the heatmap
# T, D = np.meshgrid(t_biroli_range, d_range)
# A = np.zeros_like(T)

# # Calculate ambiguity ratio for each (t, d) combination
# print("Calculating ambiguity ratio heatmap...")
# for i in tqdm(range(len(d_range)), desc="Distance"):
#     for j in range(len(t_biroli_range)):
#         A[i, j] = ambiguity_ratio(np.array([T[i, j]]), d=D[i, j])[0]

# # Create the main heatmap figure
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # --- Subplot 1: Main Heatmap ---
# ax1 = axes[0, 0]
# im1 = ax1.contourf(T, D, A, levels=50, cmap='viridis')
# cbar1 = plt.colorbar(im1, ax=ax1)
# cbar1.set_label('Ambiguity Ratio A(t,d)', fontsize=12)

# # Add contour lines for specific ambiguity levels
# contours1 = ax1.contour(T, D, A, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
#                         colors='white', linewidths=1.5, alpha=0.8)
# ax1.clabel(contours1, inline=True, fontsize=10, fmt='%.1f')

# # Mark the actual d_min and characteristic times
# ax1.axhline(y=d_min, color='red', linestyle='--', linewidth=2, alpha=0.9, 
#             label=f'd_min = {d_min:.3f}')
# ax1.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, alpha=0.9,
#             label=f't_sep = {t_C_strict:.3f}')

# ax1.set_xlabel('Time t (Biroli Scale)', fontsize=12)
# ax1.set_ylabel('Distance d', fontsize=12)
# ax1.set_title('Ambiguity Ratio A(t,d) Heatmap', fontsize=14)
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # --- Subplot 2: Cross-section at d_min ---
# ax2 = axes[0, 1]
# ambiguity_at_dmin = ambiguity_ratio(t_biroli_range, d=d_min)
# ax2.plot(t_biroli_range, ambiguity_at_dmin, 'r-', linewidth=2, label=f'A(t, d_min) where d_min={d_min:.3f}')

# # Add characteristic time markers
# ax2.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, 
#             label=f't_sep = {t_C_strict:.3f}')

# # Add horizontal reference lines
# ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='A(t) = 0.5')
# ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7, label='A(t) = 0.1')

# ax2.set_xlabel('Time t (Biroli Scale)', fontsize=12)
# ax2.set_ylabel('Ambiguity Ratio A(t)', fontsize=12)
# ax2.set_title('Cross-section: A(t) at d_min', fontsize=14)
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# # --- Subplot 3: Cross-section at t_sep ---
# ax3 = axes[1, 0]
# ambiguity_at_tsep = ambiguity_ratio(np.array([t_C_strict]), d=d_range)
# ax3.plot(d_range, ambiguity_at_tsep, 'g-', linewidth=2, label=f'A(t_sep, d) where t_sep={t_C_strict:.3f}')

# # Add d_min marker
# ax3.axvline(x=d_min, color='red', linestyle='--', linewidth=2, label=f'd_min = {d_min:.3f}')

# # Add quantile markers
# quantiles_to_mark = [0.25, 0.5, 0.75, 0.9]
# quantile_distances = torch.quantile(all_distances, torch.tensor(quantiles_to_mark).to('cuda'), dim=0).cpu().numpy()
# colors_q = plt.cm.plasma(np.linspace(0, 1, len(quantiles_to_mark)))

# for i, (q, d_q, color) in enumerate(zip(quantiles_to_mark, quantile_distances, colors_q)):
#     if d_q <= d_range.max() and d_q >= d_range.min():
#         ax3.axvline(x=d_q, color=color, linestyle=':', alpha=0.8, label=f'Q{q:.2f} = {d_q:.3f}')

# ax3.set_xlabel('Distance d', fontsize=12)
# ax3.set_ylabel('Ambiguity Ratio A(d)', fontsize=12)
# ax3.set_title('Cross-section: A(d) at t_sep', fontsize=14)
# ax3.legend()
# ax3.grid(True, alpha=0.3)

# # --- Subplot 4: Log-scale version for better detail ---
# ax4 = axes[1, 1]
# im4 = ax4.contourf(T, D, np.log10(A + 1e-10), levels=50, cmap='plasma')
# cbar4 = plt.colorbar(im4, ax=ax4)
# cbar4.set_label('log₁₀(Ambiguity Ratio)', fontsize=12)

# # Add contour lines for specific log ambiguity levels
# log_levels = np.log10([0.01, 0.1, 0.5])
# contours4 = ax4.contour(T, D, np.log10(A + 1e-10), levels=log_levels, 
#                         colors='white', linewidths=1.5, alpha=0.8)
# ax4.clabel(contours4, inline=True, fontsize=10, fmt='%.2f')

# # Mark the actual d_min and characteristic times
# ax4.axhline(y=d_min, color='red', linestyle='--', linewidth=2, alpha=0.9, label=f'd_min = {d_min:.3f}')
# ax4.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, alpha=0.9, label=f't_sep = {t_C_strict:.3f}')

# ax4.set_xlabel('Time t (Biroli Scale)', fontsize=12)
# ax4.set_ylabel('Distance d', fontsize=12)
# ax4.set_title('Log₁₀(Ambiguity Ratio) for Better Detail', fontsize=14)
# ax4.legend()
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(f"../outputs/{dataset_name}_ambiguity_ratio_comprehensive_heatmap.png", 
#             dpi=300, bbox_inches='tight')
# plt.show()

#%%
# ==============================================================================
# INTERACTIVE-STYLE HEATMAP WITH DISTANCE QUANTILES
# ==============================================================================

# # Calculate quantiles for better distance range
# quantiles_fine = np.linspace(0.01, 0.99, 50)
# distances_fine = torch.quantile(all_distances, torch.tensor(quantiles_fine).to(torch.float32).to("cuda"), dim=0).cpu().numpy()

# # Create meshgrid using quantiles instead of absolute distances
# T_fine, Q = np.meshgrid(t_biroli_range, quantiles_fine)
# A_fine = np.zeros_like(T_fine)

# print("Calculating fine-grained ambiguity ratio heatmap...")
# for i in tqdm(range(len(quantiles_fine)), desc="Quantiles"):
#     d = distances_fine[i]
#     for j in range(len(t_biroli_range)):
#         A_fine[i, j] = ambiguity_ratio(np.array([T_fine[i, j]]), d=d)[0]

# # Create the quantile-based heatmap
# plt.figure(figsize=(14, 10))

# # Main heatmap
# im = plt.contourf(T_fine, Q, A_fine, levels=50, cmap='viridis')
# cbar = plt.colorbar(im)
# cbar.set_label('Ambiguity Ratio A(t,d)', fontsize=14)

# # Add contour lines
# contours = plt.contour(T_fine, Q, A_fine, levels=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9], 
#                       colors='white', linewidths=1.2, alpha=0.8)
# plt.clabel(contours, inline=True, fontsize=10, fmt='%.2f')

# # Add characteristic time line
# plt.axvline(x=t_C_strict, color='red', linestyle='--', linewidth=3, alpha=0.9,
#             label=f't_sep = {t_C_strict:.3f}')

# # Mark special quantiles
# special_quantiles = [0.01, 0.25, 0.5, 0.75, 0.95]
# for sq in special_quantiles:
#     plt.axhline(y=sq, color='yellow', linestyle=':', alpha=0.6, linewidth=1)
#     plt.text(0.02, sq + 0.01, f'Q{sq:.2f}', color='yellow', fontweight='bold')

# plt.xlabel('Time t (Biroli Scale)', fontsize=14)
# plt.ylabel('Distance Quantile', fontsize=14)
# plt.title(f'Ambiguity Ratio A(t,d) vs Time and Distance Quantile\n({dataset_name})', fontsize=16)
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Add text annotations for interpretation
# plt.text(0.1, 0.95, 'High Ambiguity\n(Close neighbors)', 
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#          fontsize=10, ha='left')
# plt.text(2.0, 0.05, 'Low Ambiguity\n(Distant neighbors)', 
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7),
#          fontsize=10, ha='center', color='white')

# plt.tight_layout()
# plt.savefig(f"../outputs/{dataset_name}_ambiguity_quantile_heatmap.png", 
#             dpi=300, bbox_inches='tight')
# plt.show()

#%%
# ==============================================================================
# COMPREHENSIVE AMBIGUITY RATIO HEATMAP VISUALIZATION - VERSION 2
# ==============================================================================

# Calculate quantiles for better distance range (same as the interactive plot)
quantiles_fine = np.linspace(0, 1, 1000)
distances_fine = torch.quantile(all_distances, torch.tensor(quantiles_fine).to(torch.float32).to("cuda"), dim=0).cpu().numpy()

# Define the parameter space for the heatmap
t_biroli_range = np.linspace(0.01, 2.5, 100)  # Time range
d_range = np.linspace(d_min * 0.5, d_min * 3.0, 100)  # Distance range around d_min (for subplots 2 and 3)

t_C_strict = -0.5 * np.log(1 - (d_min / (2 * 2.5))**2) # From our formula
d_min_ambiguity = ambiguity_ratio(t_C_strict, d=d_min)

# Create meshgrid for quantile-based heatmap (subplots 1 and 4)
T_quantile, Q = np.meshgrid(t_biroli_range, quantiles_fine)
A_quantile = np.zeros_like(T_quantile)

# Create meshgrid for absolute distance heatmap (subplots 2 and 3)
T, D = np.meshgrid(t_biroli_range, d_range)
A = np.zeros_like(T)

# Calculate ambiguity ratio for quantile-based heatmap
print("Calculating quantile-based ambiguity ratio heatmap...")
for i in tqdm(range(len(quantiles_fine)), desc="Quantiles"):
    d = distances_fine[i]
    for j in range(len(t_biroli_range)):
        A_quantile[i, j] = ambiguity_ratio(np.array([T_quantile[i, j]]), d=d)[0]

# Calculate ambiguity ratio for absolute distance heatmap
print("Calculating absolute distance ambiguity ratio heatmap...")
for i in tqdm(range(len(d_range)), desc="Distance"):
    for j in range(len(t_biroli_range)):
        A[i, j] = ambiguity_ratio(np.array([T[i, j]]), d=D[i, j])[0]

# Create the main heatmap figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Subplot 1: Quantile-based Heatmap ---
ax1 = axes[0, 0]
im1 = ax1.contourf(T_quantile, Q, A_quantile, levels=50, cmap='plasma')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Ambiguity Ratio A(t,d)', fontsize=12)

# Add contour lines for specific ambiguity levels
contours1 = ax1.contour(T_quantile, Q, A_quantile, levels=[d_min_ambiguity, 0.1, 0.3, 0.5, 0.7, 0.9], colors='white', linewidths=1.5, alpha=0.8)
ax1.clabel(contours1, inline=True, fontsize=10, fmt='%.1f')

# Mark the actual d_min quantile and characteristic times
d_min_quantile = torch.sum(all_distances <= d_min).float() / len(all_distances)
# ax1.axhline(y=d_min_quantile.cpu().numpy(), color='red', linestyle='--', linewidth=2, alpha=0.9, label=f'd_min quantile = {d_min_quantile:.3f}')
ax1.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, alpha=0.9,label=f't_sep = {t_C_strict:.3f}')

# Mark special quantiles
special_quantiles = [0.01, 0.25, 0.5, 0.75, 0.95]
# for sq in special_quantiles:
#     ax1.axhline(y=sq, color='yellow', linestyle=':', alpha=0.6, linewidth=1)

ax1.set_xlabel('Time t (Biroli Scale)', fontsize=12)
ax1.set_ylabel('Distance Quantile', fontsize=12)
ax1.set_title('Ambiguity Ratio A(t,d) vs Distance Quantile', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Cross-section at d_min ---
ax2 = axes[0, 1]
ambiguity_at_dmin = ambiguity_ratio(t_biroli_range, d=d_min)
ax2.plot(t_biroli_range, ambiguity_at_dmin, 'r-', linewidth=2, label=f'A(t, d_min) where d_min={d_min:.3f}')

# Add characteristic time markers
ax2.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, label=f't_sep = {t_C_strict:.3f}')

# Add horizontal reference lines
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='A(t) = 0.5')
ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7, label='A(t) = 0.1')

ax2.set_xlabel('Time t (Biroli Scale)', fontsize=12)
ax2.set_ylabel('Ambiguity Ratio A(t)', fontsize=12)
ax2.set_title('Cross-section: A(t) at d_min', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Cross-section at t_sep ---
ax3 = axes[1, 0]
ambiguity_at_tsep = ambiguity_ratio(np.array([t_C_strict]), d=d_range)
ax3.plot(d_range, ambiguity_at_tsep, 'g-', linewidth=2, label=f'A(t_sep, d) where t_sep={t_C_strict:.3f}')

# Add d_min marker
ax3.axvline(x=d_min, color='red', linestyle='--', linewidth=2, label=f'd_min = {d_min:.3f}')

# Add quantile markers
quantiles_to_mark = [0.25, 0.5, 0.75, 0.9]
quantile_distances = torch.quantile(all_distances, torch.tensor(quantiles_to_mark).to('cuda'), dim=0).cpu().numpy()
colors_q = plt.cm.plasma(np.linspace(0, 1, len(quantiles_to_mark)))

for i, (q, d_q, color) in enumerate(zip(quantiles_to_mark, quantile_distances, colors_q)):
    if d_q <= d_range.max() and d_q >= d_range.min():
        ax3.axvline(x=d_q, color=color, linestyle=':', alpha=0.8, label=f'Q{q:.2f} = {d_q:.3f}')

ax3.set_xlabel('Distance d', fontsize=12)
ax3.set_ylabel('Ambiguity Ratio A(d)', fontsize=12)
ax3.set_title('Cross-section: A(d) at t_sep', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Subplot 4: Quantile-based Log-scale version for better detail ---
ax4 = axes[1, 1]
im4 = ax4.contourf(T_quantile, Q, np.log10(A_quantile + 1e-10), levels=50, cmap='viridis')
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('log₁₀(Ambiguity Ratio)', fontsize=12)

# Add contour lines for specific log ambiguity levels
log_levels = np.log10([d_min_ambiguity, 0.01, 0.1, 0.5])
contours4 = ax4.contour(T_quantile, Q, np.log10(A_quantile + 1e-10), levels=log_levels, colors='black', linewidths=1.5, alpha=0.8)
ax4.clabel(contours4, inline=True, fontsize=10, fmt='%.2f')

# Mark the actual d_min quantile and characteristic times
# ax4.axhline(y=d_min_quantile, color='red', linestyle='--', linewidth=2, alpha=0.9, label=f'd_min quantile = {d_min_quantile:.3f}')
ax4.axvline(x=t_C_strict, color='orange', linestyle='--', linewidth=2, alpha=0.9, label=f't_sep = {t_C_strict:.3f}')

# Mark special quantiles
for sq in special_quantiles:
    ax4.axhline(y=sq, color='yellow', linestyle=':', alpha=0.6, linewidth=1)

ax4.set_xlabel('Time t (Biroli Scale)', fontsize=12)
ax4.set_ylabel('Distance Quantile', fontsize=12)
ax4.set_title('Log₁₀(Ambiguity Ratio) vs Distance Quantile', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig(f"../outputs/{dataset_name}_ambiguity_ratio_comprehensive_heatmap.png", 
#             dpi=300, bbox_inches='tight')
plt.show()

#%%
# ==============================================================================
# CORRECTED CRITICAL TIME ANALYSIS
# ==============================================================================

# For each distance quantile, find the time where ambiguity drops below thresholds
thresholds = [0.01, 0.05, 0.1, 0.3, 0.5]
thresholds = np.linspace(0, 1, 1000)
critical_times_matrix = np.full((len(quantiles_fine), len(thresholds)), np.nan)

print("Calculating critical times for each quantile and threshold...")
for i, d in enumerate(tqdm(distances_fine, desc="Distances")):
    # Ensure d is a positive float, not a tensor or array element
    d_val = float(d)
    if d_val <= 0: continue
    
    ambiguity_schedule = ambiguity_ratio(t_biroli_range, d=d_val)
    
    for j, threshold in enumerate(thresholds):
        # CORRECTED LOGIC: Find all indices where ambiguity is below the threshold
        below_indices = np.where(ambiguity_schedule < threshold)[0]
        
        if len(below_indices) > 0:
            # The critical time is the LATEST time that satisfies the condition.
            last_below_idx = below_indices[-1]
            critical_times_matrix[i, j] = t_biroli_range[last_below_idx]
        # If it's never below the threshold, the value remains NaN, which is correct.

# Create the critical times visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Heatmap of critical times
im = ax1.imshow(
    critical_times_matrix, 
    aspect='auto', 
    cmap='plasma', 
    origin='lower', # Set origin to lower-left for standard plot orientation
    extent=[thresholds[0], thresholds[-1], quantiles_fine[0], quantiles_fine[-1]]
)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label('Critical Time t (Biroli Scale)', fontsize=12)
ax1.set_xlabel('Ambiguity Threshold A(t)', fontsize=12)
ax1.set_ylabel('Distance Quantile', fontsize=12)
ax1.set_title('Critical Times: When A(t,d) drops below threshold', fontsize=14)


thresholds = [0.01, 0.05, 0.1, 0.3, 0.5]
# Line plot showing how critical time varies with quantile for each threshold
for j, threshold in enumerate(thresholds):
    valid_times = critical_times_matrix[:, j]
    valid_mask = ~np.isnan(valid_times)
    if np.any(valid_mask):
        ax2.plot(quantiles_fine[valid_mask], valid_times[valid_mask], 
                 'o-', label=f'A(t) < {threshold}', alpha=0.8, markersize=4)

ax2.axhline(y=t_C_strict, color='red', linestyle='--', linewidth=2, alpha=0.9,
            label=f't_sep = {t_C_strict:.3f}')

ax2.set_xlabel('Distance Quantile', fontsize=12)
ax2.set_ylabel('Critical Time t (Biroli Scale)', fontsize=12)
ax2.set_title('Critical Time vs Distance Quantile', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# --- Run Calculations (Collapse) ---
n_prime = 200000  # Number of Monte Carlo samples for entropy estimation
batch_size = 1024 * 2

# Scan a range of time points. Adjust based on t_S_biroli.
t_scan_range = np.linspace(0.01, 0.6, 500)

t_C_biroli, f_values = calculate_collapse_time_torch(filtered_data, t_scan_range, n_prime, batch_size)


# %%

# --- Plot the Results ---
if t_C_biroli is not None:
	alpha = np.log(n_samples_total) / d_features
	f_values_normalized = f_values / alpha

	plt.figure(figsize=(10, 6))
	plt.plot(t_scan_range, f_values_normalized, "o-", label=r"$f^e(t) / \alpha$")
	plt.axvline(x=t_C_biroli, color="r", linestyle="--", label=f"Calculated $t_C = {t_C_biroli:.3f}$")
	# plt.axvline(
	# 	x=t_S_biroli_one_channel,
	# 	color="k",
	# 	linestyle="--",
	# 	label=f"Calculated $t_S (1ch) = {t_S_biroli_one_channel:.3f}$",
	# )
	plt.axvline(x=t_S_biroli, color="g", linestyle="--", label=f"Calculated $t_S = {t_S_biroli:.3f}$")
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

	plt.savefig("cifar10_collapse_plot.png")
	print("\nPlot saved to cifar10_collapse_plot.png")
	plt.show()

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
	target_logSNR = np.log((exp_m2t) / (1 - exp_m2t + eps))

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

t_C_biroli = 0.5

# 1. Instantiate your schedule
# These are the default values, you can change them if you use different ones.
custom_schedule = CosineSchedule()

# 2. Use computed Biroli times to convert them to your schedule
print("Converting characteristic times for your Custom Cosine Schedule.")
print("-" * 60)

print("Converting Speciation Time (t_S):")
t_S_your = convert_biroli_to_custom_cosine(t_S_biroli, custom_schedule)

print("\n" + "-" * 60)
print("Converting Speciation Time (t_S_one_channel):")
t_S_your_one_channel = convert_biroli_to_custom_cosine(t_S_biroli_one_channel, custom_schedule)

print("\n" + "-" * 60)
print("Converting Collapse Time (t_C):")
t_C_your = convert_biroli_to_custom_cosine(t_C_biroli, custom_schedule)

print("\n" + "-" * 60)
print("Converting Separation Time (t_C_strict):")
t_C_strict_your = convert_biroli_to_custom_cosine(t_C_strict, custom_schedule)

print("\n" + "-" * 60)
print("Converting Separation Time (t_C_stricter):")
t_C_stricter_your = convert_biroli_to_custom_cosine(t_C_stricter, custom_schedule)

print("\n" + "=" * 70)
print("                       SUMMARY OF RESULTS")
print("=" * 70)
print(f"| {'Time':<16} | {'Biroli Scale [0, ∞)':<25} | {'Your Cosine Schedule [0, 1]':<30} |")
print(f"|{'-' * 18}|{'-' * 27}|{'-' * 32}|")
print(f"| {'Speciation':<16} | {t_S_biroli:<25.4f} | {t_S_your:<30.4f} |")
print(f"| {'Speciation (1ch)':<16} | {t_S_biroli_one_channel:<25.4f} | {t_S_your_one_channel:<30.4f} |")
print(f"| {'Collapse':<16} | {t_C_biroli:<25.4f} | {t_C_your:<30.4f} |")
print(f"| {'Separation':<16} | {t_C_strict:<25.4f} | {t_C_strict_your:<30.4f} |")
print(f"| {'Separation++':<16} | {t_C_stricter:<25.4f} | {t_C_stricter_your:<30.4f} |")
print("=" * 70)

# Optional: Plot the schedule to visualize the conversion
t_range_your = np.linspace(0.0, 1.0, 4000)
logsnr_range = custom_schedule.logsnrs.numpy()

sampling_logsnrs = [11.52642455, 3.34719069, 1.96033694, 1.0598219, 0.32277325,
                    -0.37434994, -1.1182489, -2.03821651, -3.48897367, -22.60847012]
sampling_logsnrs_ddim = [11.52642455, 3.54582581, 2.18034482, 1.30449938, 0.60634166,
                         -0.02618558, -0.66133256, -1.36883547, -2.26788009, -3.70658827]

def get_t_from_logsnr(logsnr):
	closest_discrete_timestep = np.argmin(np.abs(custom_schedule.logsnrs - logsnr))
	return closest_discrete_timestep / 4000
sampling_ts = [get_t_from_logsnr(logsnr) for logsnr in sampling_logsnrs]
sampling_ts_ddim = [get_t_from_logsnr(logsnr) for logsnr in sampling_logsnrs_ddim]


logsnr_S = np.log(np.exp(-2 * t_S_biroli) / (1 - np.exp(-2 * t_S_biroli)))
logsnr_S_one_channel = np.log(np.exp(-2 * t_S_biroli_one_channel) / (1 - np.exp(-2 * t_S_biroli_one_channel)))
logsnr_C = np.log(np.exp(-2 * t_C_biroli) / (1 - np.exp(-2 * t_C_biroli)))
logsnr_C_strict = np.log(np.exp(-2 * t_C_strict) / (1 - np.exp(-2 * t_C_strict)))
logsnr_C_stricter = np.log(np.exp(-2 * t_C_stricter) / (1 - np.exp(-2 * t_C_stricter)))

plt.figure(figsize=(15, 10))
plt.plot(t_range_your, logsnr_range, label="Your schedule: logSNR vs. t_your")
plt.scatter(t_C_your, logsnr_C, color="red", zorder=5)
plt.scatter(t_S_your_one_channel, logsnr_S_one_channel, color="black", zorder=5, label="t_S (1ch)")
plt.scatter(t_S_your, logsnr_S, color="green", zorder=5)
plt.scatter(t_C_strict_your, logsnr_C_strict, color="orange", zorder=5, label="t_C strict")
plt.scatter(t_C_stricter_your, logsnr_C_stricter, color="orange", zorder=5, label="t_C stricter")
plt.axhline(
	y=logsnr_S_one_channel,
	color="black",
	linestyle="--",
	alpha=0.7,
	label=f"$t_S (1ch)$ Level (logSNR={logsnr_S_one_channel:.2f})",
)
plt.axhline(y=logsnr_S, color="green", linestyle="--", alpha=0.7, label=f"$t_S$ Level (logSNR={logsnr_S:.2f})")
plt.axhline(y=logsnr_C, color="red", linestyle="--", alpha=0.7, label=f"$t_C$ Level (logSNR={logsnr_C:.2f})")
plt.axhline(y=logsnr_C_strict, color="orange", linestyle="--", alpha=0.7, label=f"$t_C strict$ Level (logSNR={logsnr_C_strict:.2f})")
plt.axhline(y=logsnr_C_stricter, color="orange", linestyle="--", alpha=0.7, label=f"$t_C stricter$ Level (logSNR={logsnr_C_stricter:.2f})")
plt.ylim(-15, 15)

plt.axvline(x=t_S_your, color="green", linestyle="--", linewidth=1.5, label=f"$t_S = {t_S_your:.2f}$")
plt.axvline(x=t_C_your, color="red", linestyle="--", linewidth=1.5, label=f"$t_C = {t_C_your:.2f}$")
plt.axvline(x=t_C_strict_your, color="orange", linestyle="--", linewidth=1.5, label=f"$t_C strict = {t_C_strict_your:.2f}$")
plt.axvline(x=t_C_stricter_your, color="orange", linestyle="--", linewidth=1.5, label=f"$t_C stricter = {t_C_stricter_your:.2f}$")

# Add sampling points at the bottom of the plot
y_min, _ = plt.ylim()
plt.scatter(sampling_ts, [y_min + 1] * len(sampling_ts), color='blue', marker='.', zorder=10, label='Sampling Timesteps (iDDPM)')
plt.scatter(sampling_ts_ddim, [y_min + 2] * len(sampling_ts_ddim), color='brown', marker='.', zorder=10, label='Sampling Timesteps (DDIM)')


plt.xlabel("Your Normalized Time t_your [0, 1]")
plt.ylabel("log(SNR)")
plt.title(f"Mapping Biroli Times to Your Cosine Noise Schedule - {dataset_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"../outputs/{dataset_name}_time_matching_dynamical.png")
plt.show()

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

print("\n" + "-" * 60)
print("Converting Separation Time (t_C_strict):")
t_C_strict_your = convert_biroli_to_custom_cosine(t_C_strict, custom_schedule)

print("\n" + "-" * 60)
print("Converting Separation Time (t_C_stricter):")
t_C_stricter_your = convert_biroli_to_custom_cosine(t_C_stricter, custom_schedule)

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
