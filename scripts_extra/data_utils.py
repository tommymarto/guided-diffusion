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