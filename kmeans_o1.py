from memory_profiler import profile
import tracemalloc
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def read_img(img_path: str):
	return Image.open(img_path)

def show_img(img_2d: Image):
	plt.imshow(img_2d)
	plt.axis('off')
	plt.show()

def save_img(img_2d: Image, img_path: str):
	img_2d.save(img_path)

def convert_img_to_1d(img_2d: Image):
	img_np = np.array(img_2d)
	return img_np.reshape(-1, 3)

def generate_2d_img(img_2d_shape, centroids, labels):
	height, width, channels = img_2d_shape
	clusters = centroids[labels]
	return Image.fromarray(clusters.reshape((height, width, channels)).astype(np.uint8))

def count_color(img: Image) -> int:
	img_1d = convert_img_to_1d(img)
	return np.unique(img_1d, axis=0).shape[0]

BATCH_SIZE: int = 10000
EPSILON: float = 1e-5

@profile
def kmeans_o1(img_1d: np.ndarray, k_clusters: int, max_iter: int, init_centroids: str = 'random', batch_size: int = BATCH_SIZE, epsilon: float = EPSILON) -> tuple[np.ndarray, np.ndarray]:
	"""
	K-Means algorithm
	Parameters
	----------
	:param img_1d: np.ndarray with shape=(height * width, num_channels) | Original (1D) image
	:param epsilon: Checking float number
	:param batch_size: Checking batch size for dot multiplication
	k_clusters : int
		Number of clusters
	max_iter : int
		Max iterator
	init_centroids : str, default='random'
		The method used to initialize the centroids for K-means clustering
		'random' --> Centroids are initialized with random values between 0 and 255 for each channel
		'in_pixels' --> A random pixel from the original image is selected as a centroid for each cluster

	Returns
	-------
	centroids : np.ndarray with shape=(k_clusters, num_channels)
		Stores the color centroids for each cluster
	labels : np.ndarray with shape=(height * width, )
		Stores the cluster label for each pixel in the image
	"""
	n_pixels, n_channels = img_1d.shape
	img_1d = img_1d.astype(np.float32, copy=False)
	centroids: np.ndarray
	if init_centroids == 'random':
		centroids = np.random.uniform(0, 255, size=(k_clusters, n_channels)).astype(np.float32)
	elif init_centroids == 'in_pixels':
		indices = np.random.choice(n_pixels, size=k_clusters, replace=False)
		centroids = img_1d[indices].astype(np.float32)
	else:
		raise ValueError("Invalid init_centroids value")

	labels = np.zeros(n_pixels, dtype=np.int32)
	# compute distances without creating a bigger matrix: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x*c
	img_sq = np.sum(img_1d ** 2, axis=1)  # precompute ||x||^2
	cent_sq = np.sum(centroids ** 2, axis=1)  # ||c||^2
	for _ in range(max_iter):
		new_labels = np.zeros(n_pixels, dtype=np.int32)
		for start in range(0, n_pixels, batch_size):
			end = min(start + batch_size, n_pixels)
			batch_img = img_1d[start:end]
			batch_img_sq = img_sq[start:end][:, None]

			dot_product = batch_img @ centroids.T
			distances_sq = batch_img_sq + cent_sq[None, :] - 2 * dot_product
			new_labels[start:end] = np.argmin(distances_sq, axis=1)

		# check for label convergence
		if np.array_equal(new_labels, labels):
			break

		labels[:] = new_labels

		sums = np.zeros((k_clusters, n_channels), dtype=np.float32)
		counts = np.bincount(new_labels, minlength=k_clusters)
		np.add.at(sums, new_labels, img_1d)
		# only update valid centroids (only to avoid division by zero)
		valid = counts > 0
		cent_sq[valid] = np.sum(centroids[valid] ** 2, axis=1)
		centroids[valid] = sums[valid] / counts[valid][:, None]

		# check centroid convergence
		if np.all(np.abs(centroids[valid] - sums[valid] / counts[valid][:, None]) < epsilon): # maybe there is a better way to do this? EG: almost equal?
			break

	return centroids.astype(np.uint8), labels

tracemalloc.start()
img_path = 'cat.jpg'
img = read_img(img_path)
img_1d = convert_img_to_1d(img)
img_centroids, img_labels = kmeans_o1(img_1d, k_clusters=2000, max_iter=100, init_centroids='in_pixels')
compressed_img = generate_2d_img(np.array(img).shape, img_centroids, img_labels)
show_img(compressed_img)
save_img(compressed_img, 'cat_1.png')
current, peak = tracemalloc.get_traced_memory()  # Get current and peak memory usage
tracemalloc.stop()  # Stop tracing memory

print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")