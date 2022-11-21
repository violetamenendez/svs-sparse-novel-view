import torch

def gradient_x(img):
	gx = img[:,:,:-1,:] - img[:,:,1:,:]
	return gx

def gradient_y(img):
	gy = img[:,:-1,:,:] - img[:,1:,:,:]
	return gy

def get_disparity_smoothness(disp, img):
	"""Disparity smoothness loss

	Smoothness of disparity weighted by the smoothness of the image
	Similar to Monodepth (https://github.com/mrharicot/monodepth)
	"""
	disp_gradients_x = gradient_x(disp)
	disp_gradients_y = gradient_y(disp)

	image_gradients_x = gradient_x(img)
	image_gradients_y = gradient_y(img)

	weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 3, keepdim=True))
	weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 3, keepdim=True))

	smoothness_x = torch.mean(torch.abs(disp_gradients_x) * weights_x)
	smoothness_y = torch.mean(torch.abs(disp_gradients_y) * weights_y)
	return smoothness_x + smoothness_y

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:])) + \
        torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]))
    return loss

def distortion_loss(ray_weights, t_vals):
	"""Calculate distortion loss

	From Mip-NeRF 360 (https://github.com/google-research/multinerf)
	\mathcal{L}_{dist}(\mathbf{t}, \mathbf{w}) =
		\sum_{i,j} w_{i} w_{j} \left| \frac{t_{i} + t_{i+1}}{2} - \frac{t_{j} + t_{j+1}}{2} \right|
		+ \frac{1}{3}\sum _{i} w_{i}^{2}( t_{i+1} - t_{i}))

	Args:
		ray_weights: [N, N_rays, N_samples] alpha compositing weights assigned to each sample along a ray
		t_vals:      [N, N_samples] sample positions along a ray. Normalised.
	"""

	# Product of every pair of point weights
	N, N_rays, N_samples = ray_weights.shape
	w_expanded = ray_weights[...,None].expand(-1, -1, N_samples, N_samples)
	w_transpose = w_expanded.transpose(-2, -1)
	w_pairs = w_expanded * w_transpose # [N, N_rays, N_samples, N_samples]

	# Distances between all pairs of interval midpoints
	t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
	pairs_interval_midpoints = torch.abs(t_mids[...,None] - t_mids)

	# Weighted distances
	# \sum_{i,j} w_{i} w_{j} \left| \frac{t_{i} + t_{i+1}}{2} - \frac{t_{j} + t_{j+1}}{2} \right|
	weighted_dist = 0.5 * torch.sum(w_pairs[..., :-1, :-1] * pairs_interval_midpoints, axis=[-1,-2])

	# Weighted size of each individual interval
	w_square = ray_weights * ray_weights
	t_dists = t_vals[..., 1:] - t_vals[..., :-1]
	individual_interval_size = (1/3) * torch.sum(w_square[..., :-1] * t_dists, axis=-1)

	loss = torch.sum(weighted_dist + individual_interval_size)

	return loss
