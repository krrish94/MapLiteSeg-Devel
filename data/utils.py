import os
from PIL import Image
from natsort import natsorted
import numpy as np
import imageio

import torch
import torchvision.transforms as transforms


def get_files(folder, name_filter=None, extension_filter=None):
	"""Helper function that returns the list of files in a specified folder
	with a specified extension.

	Keyword arguments:
	- folder (``string``): The path to a folder.
	- name_filter (```string``, optional): The returned files must contain
	this substring in their filename. Default: None; files are not filtered.
	- extension_filter (``string``, optional): The desired file extension.
	Default: None; files are not filtered

	"""
	if not os.path.isdir(folder):
		raise RuntimeError("\"{0}\" is not a folder.".format(folder))

	# Filename filter: if not specified don't filter (condition always true);
	# otherwise, use a lambda expression to filter out files that do not
	# contain "name_filter"
	if name_filter is None:
		# This looks hackish...there is probably a better way
		name_cond = lambda filename: True
	else:
		name_cond = lambda filename: name_filter in filename

	# Extension filter: if not specified don't filter (condition always true);
	# otherwise, use a lambda expression to filter out files whose extension
	# is not "extension_filter"
	if extension_filter is None:
		# This looks hackish...there is probably a better way
		ext_cond = lambda filename: True
	else:
		ext_cond = lambda filename: filename.endswith(extension_filter)

	filtered_files = []

	# Explore the directory tree to get files that contain "name_filter" and
	# with extension "extension_filter"
	for path, _, files in os.walk(folder):
		files.sort()
		for file in files:
			if name_cond(file) and ext_cond(file):
				full_path = os.path.join(path, file)
				filtered_files.append(full_path)

	return filtered_files


def maplite_loader(data_path, label_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.], ignore_ring=False, \
	random_mask=False):
	"""Loads a sample and label image given their path as PIL images. (maplite classes)

	Keyword arguments:
	- data_path (``string``): The filepath to the image.
	- label_path (``string``): The filepath to the ground-truth image.
	- color_mean (``list``): R, G, B channel-wise mean
	- color_std (``list``): R, G, B channel-wise stddev
	- ignore_ring (``bool``): Sets the ring dimension (the third channel) to zero
	- random_mask (``bool``): Randomly masks a section of the image

	Returns the image and the label as PIL images.

	"""

	# Load image
	data = np.array(imageio.imread(data_path))
	# Reshape data from H x W x C to C x H x W
	data = np.moveaxis(data, 2, 0)
	if ignore_ring:
		data[2,:,:] = 0
	# Define normalizing transform
	normalize = transforms.Normalize(mean=color_mean, std=color_std)
	# Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
	data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

	# Load label
	label = np.array(imageio.imread(label_path)).astype(np.uint8)

	# Randomly mask a patch
	if random_mask:
		# Left edge of the mask (col)
		left_edge = np.random.randint(0,90)
		# Top edge of the mask (row)
		top_edge = np.random.randint(0,100)
		# Mask width
		width = np.random.randint(1,50)
		# Mask height
		height = np.random.randint(1,50)
		label[top_edge:top_edge+height,left_edge:left_edge+width] = 0

	return data, label


def create_label_image(output, color_palette):
	"""Create a label image, given a network output (each pixel contains class index) and a color palette.

	Args:
	- output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
	corresponding to the class label of that pixel.
	- color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
	"""
	
	label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
	for idx, color in enumerate(color_palette):
		label_image[output==idx] = color
	return label_image


def remap(image, old_values, new_values):
	assert isinstance(image, Image.Image) or isinstance(
		image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
	assert type(new_values) is tuple, "new_values must be of type tuple"
	assert type(old_values) is tuple, "old_values must be of type tuple"
	assert len(new_values) == len(
		old_values), "new_values and old_values must have the same length"

	# If image is a PIL.Image convert it to a numpy array
	if isinstance(image, Image.Image):
		image = np.array(image)

	# Replace old values by the new ones
	tmp = np.zeros_like(image)
	for old, new in zip(old_values, new_values):
		# Since tmp is already initialized as zeros we can skip new values
		# equal to 0
		if new != 0:
			tmp[image == old] = new

	return Image.fromarray(tmp)


def enet_weighing(dataloader, num_classes, c=1.02):
	"""Computes class weights as described in the ENet paper:

		w_class = 1 / (ln(c + p_class)),

	where c is usually 1.02 and p_class is the propensity score of that
	class:

		propensity_score = freq_class / total_pixels.

	References: https://arxiv.org/abs/1606.02147

	Keyword arguments:
	- dataloader (``data.Dataloader``): A data loader to iterate over the
	dataset.
	- num_classes (``int``): The number of classes.
	- c (``int``, optional): AN additional hyper-parameter which restricts
	the interval of values for the weights. Default: 1.02.

	"""
	class_count = 0
	total = 0
	for _, label in dataloader:
		label = label.cpu().numpy()

		# Flatten label
		flat_label = label.flatten()

		# Sum up the number of pixels of each class and the total pixel
		# counts for each label
		class_count += np.bincount(flat_label, minlength=num_classes)
		total += flat_label.size

	# Compute propensity score and then the weights for each class
	propensity_score = class_count / total
	class_weights = 1 / (np.log(c + propensity_score))

	return class_weights


def median_freq_balancing(dataloader, num_classes):
	"""Computes class weights using median frequency balancing as described
	in https://arxiv.org/abs/1411.4734:

		w_class = median_freq / freq_class,

	where freq_class is the number of pixels of a given class divided by
	the total number of pixels in images where that class is present, and
	median_freq is the median of freq_class.

	Keyword arguments:
	- dataloader (``data.Dataloader``): A data loader to iterate over the
	dataset.
	whose weights are going to be computed.
	- num_classes (``int``): The number of classes

	"""
	class_count = 0
	total = 0
	for _, label in dataloader:
		label = label.cpu().numpy()

		# Flatten label
		flat_label = label.flatten()

		# Sum up the class frequencies
		bincount = np.bincount(flat_label, minlength=num_classes)

		# Create of mask of classes that exist in the label
		mask = bincount > 0
		# Multiply the mask by the pixel count. The resulting array has
		# one element for each class. The value is either 0 (if the class
		# does not exist in the label) or equal to the pixel count (if
		# the class exists in the label)
		total += mask * flat_label.size

		# Sum up the number of pixels found for each class
		class_count += bincount

	# Compute the frequency and its median
	freq = class_count / total
	med = np.median(freq)

	return med / freq
