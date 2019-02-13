"""
Utils functions to help visualize stuff
"""

import argparse
import imageio
# import matplotlib
# matplotlib.use('Qt4Agg')
# import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import _pickle as pkl
import os
import scipy.stats as stats


# 		    		Mean 				Stddev
# z:         -0.05088775275065709 	0.38221495666871064
# intensity: 11.663070406395152 	43.84871921181538
# ring:      3.3788556751238974 	12.618955142552485
CHANNEL_MEAN = [-0.0501, 11.6631, 3.3789]
CHANNEL_STD = [0.3822, 43.8487, 12.6190]


# Translation of John Cook's C++ code for the Welford's algorithm
class RunningStats(object):
	"""Class to compute running mean, stddev """

	def __init__(self):
		self.count = 0
		self.old_mean = 0.
		self.new_mean = 0.
		self.old_std = 0.
		self.new_std = 0.

	def push(self, X):
		for x in X:
			self.count += 1
			if self.count == 1:
				self.old_mean = self.new_mean = x
				self.old_std = 0
			else:
				self.new_mean = self.old_mean + (x - self.old_mean)/self.count
				self.new_std = self.old_std + (x - self.old_mean) * (x - self.new_mean)
				self.old_mean = self.new_mean
				self.old_std = self.new_std

	def numDataValues(self):
		return self.count

	def mean(self):
		return self.new_mean if self.count > 0 else 0.

	def variance(self):
		return self.new_std/(self.count-1) if self.count > 1 else 0.

	def stddev(self):
		return np.sqrt(self.variance())


def flatten_pc(pc, features, sz=(1,1), statistics=None):
	"""Calculates a 2D voxel grid from a pointcloud"""    
	axes=['x','y']
	statistics = ['mean' for i in range(len(features))] if not statistics else statistics
	xy=np.vstack(pc[ax] for ax in axes).T
	sz=(1,1)
	# mins=xy.min(0)
	# maxs=xy.max(0)
	# Magic numbers, estimated by inspecting several pointclouds
	mins = [-120, -120]
	maxs = [120, 120]
	bins=[np.arange(mins[i], maxs[i]+sz[i], sz[i]) for i in range(len(axes))]
	N=[len(b) for b in bins]
	
	V=[stats.binned_statistic_2d(pc['x'], pc['y'], pc[f], statistic=statistics[i], bins=bins) \
						for i,f in enumerate(features)]
	return nan_to_zero(np.stack([v.statistic for v in V],2))


def nan_to_zero(arr):
	"""Sets all NaNs in numpy array to 0 """
	arr[np.isnan(arr)] = 0
	return arr


# Define a custom argparse.Action subclass to validate filepaths
def file_exists(filename):
	if not os.path.exists(fileName):
		raise RuntimeError(filename + ' does not exist.')
	return value


if __name__ == '__main__':

	# Command-line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('-pkl', type=file_exists, dest='pickle_file', \
		help='Path to MapLite annotated pkl file, to create train/test data from.')
	parser.add_argument('-outDir', type=file_exists, dest='target_dir', help='Directory to store generated data.')
	args = parser.parse_args()


	all_scans = pd.read_pickle(args.pickle_file)

	# Create directories to store the image and label, if they do not already exist
	image_dir = os.path.join(args.target_dir, 'seg_data')
	label_dir = os.path.join(args.target_dir, 'seg_label')
	if not os.path.isdir(args.target_dir):
		os.makedirs(args.target_dir)
	if not os.path.isdir(image_dir):
		os.makedirs(image_dir)
		os.makedirs(label_dir)

	# Number of scans
	num_scans = len(all_scans.index)

	# # Helper objects to compute running mean and stddev
	# stats_z = RunningStats()
	# stats_intensity = RunningStats()
	# stats_ring = RunningStats()

	for i in range(num_scans):
		print(i)
		scan = all_scans.iloc[i]['scan']
		road = all_scans.iloc[i]['is_road_truth']
		nan = all_scans.iloc[i]['nan']
		pc = np.array(scan,dtype={'names':('x','y','z','intensity','ring'), \
			'formats':('f8','f8','f8','<u4','b')})
		pc = append_fields(pc, 'road', road, dtypes='?')
		pc = append_fields(pc, 'nan', nan, dtypes='?')
		pc = pc[pc['nan'] != True] # Remove NaNs

		# Compute features from pointcloud
		feat = flatten_pc(pc, ['z','intensity','ring'])
		
		# # Update statistics
		# stats_z.push(np.ndarray.flatten(feat[:,:,0]))
		# stats_intensity.push(np.ndarray.flatten(feat[:,:,1]))
		# stats_ring.push(np.ndarray.flatten(feat[:,:,2]))

		# Channel-wise normalization
		for j in range(3):
			feat[:,:,j] = (feat[:,:,j] - CHANNEL_MEAN[j]) / CHANNEL_STD[j]
			feat[:,:,j] = (255 * feat[:,:,j])
		feat = feat.astype(np.uint8)

		# Create label png
		label = flatten_pc(pc, ['road'])
		label_img = np.zeros(label.shape, dtype=np.uint8)
		label_img[np.where(label>0)] = 2
		label_img[np.where(label==0)] = 1
		label_img = label_img[:,:,0]

		# Write images
		imageio.imwrite(os.path.join(image_dir, str(i).zfill(4) + '.png'), feat)
		imageio.imwrite(os.path.join(label_dir, str(i).zfill(4) + '.png'), label_img)

	# print(stats_z.mean(), stats_z.stddev())
	# print(stats_intensity.mean(), stats_intensity.stddev())
	# print(stats_ring.mean(), stats_ring.stddev())
