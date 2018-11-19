import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imread
import os
from scipy import misc
from glob import glob
import math
from pathlib import Path
from skimage.transform import rescale

def plot_points(imagename):
	iname = "images/" + imagename + ".jpg"
	img = plt.imread(iname)
	number_of_points = 4

	# load from saved
	my_file = Path("images/" + imagename + "_coordinates")
	if my_file.is_file():
		return np.loadtxt("images/" + imagename + "_coordinates")
	plt.imshow(img)
	coordinates = np.array(plt.ginput(n = number_of_points, show_clicks = True))
	# print(coordinates)
	np.savetxt("images/" + imagename + "_coordinates", coordinates)
	return coordinates

def computeH(im1_pts,im2_pts):
	#suggestion from piazza
	#https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1289595#1289595
	A = []
	for point, pointprime in zip(im1_pts, im2_pts):
		x, y = point
		xprime, yprime = pointprime
		A.append([x, y, 1, 0, 0, 0, -x*xprime, -y*xprime])
		A.append([0, 0, 0, x, y, 1, -x*yprime, -y*yprime])
	b = im2_pts.flatten()

	h = np.linalg.lstsq(A,b.T)[0]
	# print(np.linalg.lstsq(A,b))
	h = np.append(h, np.array([1]))
	h = h.reshape((3,3))
	return h

def getWarpCorners(im,H):
	# print(H.shape)
	height = im.shape[0]
	width = im.shape[1]
	corners = [[0, 0, 1], [0, height - 1, 1], [width - 1, 0, 1], [width - 1, height - 1, 1]]
	warped_corners = []

	for c in corners:
		new = H.dot(c)
		warped_corners.append(new/new[2])

	print(warped_corners)
	return warped_corners

def warpImage(im,H):
	height = im.shape[0]
	width = im.shape[1]
	warped_corners = np.array(getWarpCorners(im,H)).astype(int)

	minx = np.amin(warped_corners[:,1].flatten())
	maxx = np.amax(warped_corners[:,1].flatten())
	miny = np.amin(warped_corners[:,0].flatten())
	maxy = np.amax(warped_corners[:,0].flatten())

	new_width = maxx - minx + 1
	new_height = maxy - miny + 1
	warped_image_shape = (new_width, new_height, 3) 
	shiftx = -minx
	shifty = -miny
	warped_image = np.zeros(warped_image_shape)

	for x in range(0, new_width):
		x_dst = np.full(new_height, x)
		y_dst = np.arange(new_height)
		ones = np.ones(new_height)
		source = np.linalg.inv(H).dot([y_dst + miny, x_dst + minx, ones])
		source[0] = source[0]/source[2]
		source[1] = source[1]/source[2]
		warped_image[x_dst, y_dst, :] = im[np.clip(source[1].astype(int), 0, height-1), np.clip(source[0].astype(int),0,width-1), :]

	return warped_image, shiftx, shifty

def mosaic(im, targetim, shiftx, shifty):
	im_height, im_width = im.shape[:2]
	target_height, target_width = targetim.shape[:2]

	mos_height = max(target_height + shiftx, im_height)
	mos_width = max(target_width + shifty, im_width)

	mos = np.zeros((mos_height, mos_width, 3))

	mos[shiftx:target_height + shiftx, shifty:target_width + shifty, :] = targetim[:, :, :]

	mos[:im_height, :im_width,:] = im[:,:,:]

	return mos

def main():
	im1name = "left"
	im2name = "right"
	iname = "images/" + im1name + ".jpg"
	img1 = plt.imread(iname)
	# img1 = rescale(img1, 1.0 / 2.0)
	iname = "images/" + im2name + ".jpg"
	img2 = plt.imread(iname)
	# img2 = rescale(img2, 1.0 / 2.0)
	coor1 = plot_points(im1name)
	coor2 = plot_points(im2name)

	H = computeH(coor1, coor2)

	warped_image, shiftx, shifty = warpImage(img1, H)
	imsave("images/" + im1name+im2name+".jpg", warped_image)
	mos = mosaic(warped_image, img2, shiftx, shifty)
	imsave("images/" + im1name+im2name+"mosaic.jpg", mos)

	rectify("tile")
	rectify("road")

	t = "road"
	iname = "images/" + t + ".jpg"
	img2 = plt.imread(iname)
	warped_image, shiftx, shifty = warpImage(img1, H)

def rectify(name):
	iname = "images/" + name + ".jpg"
	img = plt.imread(iname)
	coor1 = plot_points(name)
	coor2 = plot_points(name + "2")
	H = computeH(coor1, coor2)
	warped_image, shiftx, shifty = warpImage(img, H)
	imsave("images/" + name+"rectify.jpg", warped_image)

main()