import json
import typing as tp
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def histogram_find_cuts(nbins: int) -> np.ndarray:
    """Sequence of evenly-spaced limits of each bin (e.g. [0.0, 85.0, 170.0, 255.0] for 3 bins)."""
    return np.arange(nbins + 1) * 255 / nbins


def histogram_count_values(image: np.ndarray, nbins: int) -> np.ndarray:
    """Creates a histogram of a grayscale image."""
    size_x = image.shape[0]
    size_y = image.shape[1]
    hist = np.zeros(nbins)  # Variable to store the histogram, initialized at 0.
    for i in range(size_x):
        for j in range(size_y):
            value = image[i, j]
            discretized_value = int(value * nbins / 255)
            hist[discretized_value] += 1
    return hist


def histogram_plot(image: np.ndarray, nbins) -> None:
    """Plots a histogram of a grayscale image."""
    cuts = histogram_find_cuts(nbins=nbins)
    values = histogram_count_values(image, nbins=nbins)

    centers = (cuts[:-1] + cuts[1:]) / 2
    plt.bar(centers, values, align='center', width=cuts[1]-cuts[0])
    if len(cuts) <= 30:
        plt.xticks(cuts)
    plt.show()


def negative(image: np.ndarray) -> np.ndarray:
    """Returns the negative of a grayscale image in [0, 255]."""
    return 255 - image


def log_transform(image: np.ndarray) -> np.ndarray:
    """Returns the log transformation of a grayscale image."""
    return np.log(image + 1) / np.log(256) * 255


def exp_transform(image: np.ndarray) -> np.ndarray:
    """Returns the exp transformation of a grayscale image, which should invert the log transformation."""
    return np.exp(image / 255 * np.log(256)) - 1


def gamma_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """Returns the gamma transformation of a grayscale image."""
    return np.power(image / 255, gamma) * 255


def windowing(image: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    """Linear normalization assigning values lower or equal to lower_threshold to 0, and values greater or equal to upper_threshold to 255."""
    out = (image - lower_threshold) / (upper_threshold - lower_threshold)
    out[out < 0] = 0
    out[out > 1] = 1
    return out*255


def minmax_normalization(image: np.ndarray) -> np.ndarray:
    """Linear normalization assigning the lowest value to 0 and the highest value to 255."""
    return windowing(image, np.min(image), np.max(image))


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram equalization."""
    hist = histogram_count_values(image, nbins=256)
    cumhist = np.cumsum(hist)
    mapping = minmax_normalization(cumhist)
    return mapping[image.astype('uint8')]


def clahe(image: np.ndarray, clip_limit=5.0, grid_size=(4, 4)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization."""
    image = image.astype('uint8')   # Ensure that the image is of type uint8
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def additive_white_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Adds additive white Gaussian noise to an image."""
    return image + np.random.normal(0, std, image.shape)


def uniform_multiplicative_noise(image: np.ndarray, a: float, b: float) -> np.ndarray:
    """Adds uniform multiplicative noise to an image."""
    return image * np.random.uniform(a, b, image.shape)


def salt_and_pepper_noise(image: np.ndarray, p: float) -> np.ndarray:
    """Adds salt and pepper noise to an image."""
    image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.random.uniform(0, 1) < p:
                if np.random.uniform(0, 1) <= 0.5:
                    image[i, j] = 0
                else:
                    image[i, j] = 255
    return image


def shot_noise(image: np.ndarray) -> np.ndarray:
    """Add shot noise to an image."""
    return np.random.poisson(image)


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Returns the convolution of one image with a specific kernel (with no padding)."""
    img_sz_x, img_sz_y = image.shape
    krn_sz_x, krn_sz_y = kernel.shape
    out_sz_x = img_sz_x - krn_sz_x + 1  # Why?
    out_sz_y = img_sz_y - krn_sz_y + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE:
            #   ...
            for k in range(krn_sz_x):
                for l in range(krn_sz_y):
                    out[i, j] += image[i + (krn_sz_x-1) - k, j + (krn_sz_y-1) - l] * kernel[k, l]
    return out


def kernel_squared_mean_filter(size: Tuple[int, int]) -> np.ndarray:
    """Returns a kernel the of given size for the mean filter."""
    return np.ones(shape=size, dtype=np.float32) / np.prod(size)


def kernel_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter."""
    kernel = np.zeros(shape=size, dtype=np.float32)
    for i in range(size[0]):
        for j in range(size[1]):
            kernel[i, j] = np.exp(-((i-size[0]//2)**2 + (j-size[1]//2)**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)
    return kernel


def kernel_sharpening(kernel_smoothing: np.ndarray, alpha: float) -> np.ndarray:
    """Returns a kernel for sharpening the image."""
    sz = kernel_smoothing.shape
    kernel_impulse = np.zeros_like(kernel_smoothing)
    kernel_impulse[sz[0]//2, sz[1]//2] = 1
    kernel_detail = kernel_impulse - kernel_smoothing
    kernel_sharpening = kernel_impulse + alpha * kernel_detail
    return kernel_sharpening


def kernel_horizontal_derivative() -> np.ndarray:
    """Returns a 3x1 kernel for the horizontal derivative using first order central difference coefficients. """
    return np.array([[1/2, 0, -1/2]])


def kernel_vertical_derivative() -> np.ndarray:
    """Returns a 1x3 kernel for the vertical derivative using first order central difference coefficients. """
    return kernel_horizontal_derivative().transpose()


def kernel_sobel_horizontal() -> np.ndarray:
    """Returns the sobel operator for horizontal derivatives. """
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)/8
    return sobel


def kernel_sobel_vertical() -> np.ndarray:
    """Returns the sobel operator for vertical derivatives. """
    return kernel_sobel_horizontal().transpose()


def kernel_LoG_filter() -> np.ndarray:
    """Returns a 3x3 kernel for the Laplacian of Gaussian filter."""
    log = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return log


def median_filter(image: np.ndarray, filter_size: Tuple[int, int]) -> np.ndarray:
    """Returns an image after applying the median filter of the given size."""
    img_sz_x, img_sz_y = image.shape
    out_sz_x = img_sz_x - filter_size[0] + 1  # Why?
    out_sz_y = img_sz_y - filter_size[1] + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            values = image[i:i + (filter_size[0]-1), j:j + (filter_size[1]-1)]
            out[i, j] = np.median(values)
    return out


def max_pooling(image: np.ndarray, pool_size: Tuple[int, int]) -> np.ndarray:
    """Returns an image after applying the max pooling of the given size."""
    img_sz_x, img_sz_y = image.shape
    out_sz_x = img_sz_x//pool_size[0]   # Why?
    out_sz_y = img_sz_y//pool_size[1]   # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            values = image[i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1]]
            out[i, j] = np.max(values)
    return out


def binarize_by_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    return (img >= threshold)*255


def binarize_by_otsu(img: np.ndarray) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    otsu_threshold = 0
    lowest_criteria = np.inf
    for threshold in range(255):
        thresholded_im = img >= threshold
        # compute weights
        weight1 = np.sum(thresholded_im) / img.size
        weight0 = 1 - weight1

        # if one the classes is empty, that threshold will not be considered
        if weight1 != 0 and weight0 != 0:
            # compute criteria, based on variance of these classes
            var0 = np.var(img[thresholded_im == 0])
            var1 = np.var(img[thresholded_im == 1])
            otsu_criteria = weight0 * var0 + weight1 * var1

            if otsu_criteria < lowest_criteria:
                otsu_threshold = threshold
                lowest_criteria = otsu_criteria

    return binarize_by_thresholding(img, otsu_threshold)


def binarize_by_dithering(img: np.ndarray) -> np.ndarray:
    """Returns a binary image by applying the Floyd–Steinberg dithering algorithm to a grayscale image."""
    # Add one extra row to avoid dealing with "corner cases" in the loop.
    padded_img = np.zeros(shape=(img.shape[0] + 1, img.shape[1] + 1), dtype=img.dtype)
    padded_img[:-1, :-1] = img
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value = padded_img[i, j]
            if value > 127:
                out[i, j] = 255
            error = value - out[i, j]
            padded_img[i, j + 1] += error * 7 / 16
            padded_img[i + 1, j - 1] += error * 3 / 16
            padded_img[i + 1, j] += error * 5 / 16
            padded_img[i + 1, j + 1] += error * 1 / 16
    return out


def label_connected_components(binary_img: np.ndarray) -> np.ndarray:
    """Returns a labeled version of the image, where each connected component is assigned a different label."""
    label_img = np.zeros_like(binary_img, dtype=np.uint16)
    collisions = {}     # { label: min_label_in_same_CC }
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i, j]:
                previous_labels = []
                # We use a 4-neighbour connectivity to find out previous labelled pixels
                if i >= 1 and label_img[i-1, j]:
                    previous_labels.append(label_img[i-1, j])
                if j >= 1 and label_img[i, j-1]:
                    previous_labels.append(label_img[i, j-1])

                if len(previous_labels) == 0:
                    # No labelled neighbours: create a new label
                    label_img[i, j] = np.max(label_img) + 1
                elif len(previous_labels) == 1:
                    # One labelled neighbour: use their label
                    label_img[i, j] = min(previous_labels)
                else:
                    # Multiple labelled neighbours
                    # Find minimum label in current connected component.
                    representative_label = min(previous_labels)
                    for label in previous_labels:
                        if label in collisions:
                            representative_label = min(representative_label, collisions[label])
                    # Assign current pixel and update collisions dictionary.
                    label_img[i, j] = representative_label
                    for label in previous_labels:
                        collisions[label] = representative_label

    # Make collision dictionary transitive.
    for label in range(np.max(label_img)):  # Ordered lookup is important here.
        if label in collisions:
            representative = collisions[label]
            # If representative is not a root, find its root.
            if representative in collisions:
                collisions[label] = collisions[representative]

    # Replace labels with their representatives.
    for label, min_label_in_same_cc in collisions.items():
        label_img[label_img == label] = min_label_in_same_cc
    return label_img


def binarize_by_hysteresis(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a hysteresis operation."""
    out = np.zeros_like(img)
    binary_img = binarize_by_thresholding(img, low_threshold)
    _, label_img = cv2.connectedComponents(binary_img.astype('uint8'))
    labels = np.unique(label_img)
    for label in labels:
        if label == 0:
            pass
        elif np.any(img[label_img == label] >= high_threshold):
            out[label_img == label] = 255
    return out


def object_area(binary_img: np.ndarray) -> np.ndarray:
    """Returns the area of one object, passed as a binary image that contains only one connected component."""
    return np.sum(binary_img)


def object_centroid(binary_img: np.ndarray) -> tp.Tuple[float, float]:
    """Returns the centroid of one object, passed as a binary image that contains only one connected component."""
    centroid_x = 0
    centroid_y = 0
    # We will iterate over all pixels (which is simple but very slow... it could be improved by vectorizing the operation)
    for x in range(binary_img.shape[0]):
        for y in range(binary_img.shape[1]):
            centroid_x += x * binary_img[x, y]
            centroid_y += y * binary_img[x, y]
    m00 = object_area(binary_img)
    centroid_x = centroid_x/m00
    centroid_y = centroid_y/m00
    return centroid_x, centroid_y


def largest_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the largest connected component."""
    _, label_img = cv2.connectedComponents(binary_img.astype('uint8'))
    largest_object_label = -1
    largest_object_pixels = 0
    for label in range(np.max(label_img)+1):
        if label == 0:
            pass
        else:
            area = object_area(label_img == label)
            if area > largest_object_pixels:
                largest_object_pixels = area
                largest_object_label = label
    return (label_img == largest_object_label).astype('uint8') * 255


def most_centered_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the most centered connected component."""
    _, label_img = cv2.connectedComponents(binary_img.astype('uint8'))
    object_distance_to_center = np.inf
    object_label = 0
    for label in range(np.max(label_img)+1):
        if label == 0:
            pass
        else:
            x, y = object_centroid(label_img == label)
            d = (x-binary_img.shape[0]/2)**2 + (y-binary_img.shape[1]/2)**2
            if d < object_distance_to_center:
                object_distance_to_center = d
                object_label = label
    return (label_img == object_label).astype('uint8') * 255


def dilation(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the dilation of the binary/grayscale image with the given structuring element."""
    return cv2.dilate(img.astype('uint8'), structuring_element.astype('uint8'))


def erosion(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the erosion of the binary/grayscale image with the given structuring element."""
    return cv2.erode(img.astype('uint8'), structuring_element.astype('uint8'))


def opening(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the opening of the binary/grayscale image with the given structuring element."""
    return dilation(erosion(img, structuring_element), np.flip(structuring_element))


def closing(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the closing of the binary/grayscale image with the given structuring element."""
    return erosion(dilation(img, structuring_element), np.flip(structuring_element))


def morphological_gradient(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological gradient of the binary/grayscale image with the given structuring element."""
    return dilation(img, structuring_element) - erosion(img, structuring_element)


def morphological_skeleton(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological skeleton of the binary/grayscale image considering Lantuéjoul's method."""
    # Iteratively erode the image (until there are no more pixels)
    eroded_imgs = []
    current_img = img
    while np.any(current_img):
        eroded_imgs.append(current_img)
        eroded = erosion(current_img, structuring_element)
        if np.all(eroded == current_img):
            break
        else:
            current_img = eroded

    skeleton = np.zeros_like(img)
    for eroded_img in eroded_imgs:
        skeleton += eroded_img - opening(eroded_img, structuring_element)
    return skeleton


def region_growing_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a region-growing method."""
    # Smooth image to improve results.
    img_gray = img_gray.astype('float')
    img_gray = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
    # Initialize mask.
    segmentation = np.zeros_like(img_gray)

    pixels_to_process = [seed_pixel]
    while pixels_to_process:
        pixel = pixels_to_process.pop()
        segmentation[pixel] = 1

        candidates = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1)]
        # Filter out pixels that are out of bounds
        candidates = [c for c in candidates
                      if 0 <= c[0] < img_gray.shape[0] and 0 <= c[1] < img_gray.shape[1]]
        # Filter out pixels that are already in the segmentation
        candidates = [c for c in candidates
                      if segmentation[c] == 0]
        # Filter out pixels that are too white
        candidates = [c for c in candidates
                      if img_gray[c] < 150]
        # Filter out pixels that are too different
        candidates = [c for c in candidates if
                      np.abs(img_gray[c] - img_gray[pixel]) < 50]
        # Add neighbours to the list of pixels to process
        pixels_to_process = candidates + pixels_to_process

        if np.sum(segmentation) > 0.1 * img_gray.size:
            break

    return segmentation


def segmentation_by_watershed(img_bgr: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a watershed method."""
    img_smoothed = cv2.GaussianBlur(img_bgr, ksize=(5, 5), sigmaX=0)

    #   (0 for unknown pixels, 1 for background pixels, 2 for foreground pixels)
    markers = np.ones(img_bgr.shape[:2], dtype=np.int32)
    offset = 30
    sh = img_bgr.shape
    markers[max(seed_pixel[0]-offset, 0):min(seed_pixel[0]+offset, sh[0]), max(seed_pixel[1]-offset, 0):min(seed_pixel[1]+offset, sh[1])] = 0
    markers[seed_pixel] = 2
    watshd = cv2.watershed(img_smoothed, markers=markers)
    _, axs = plt.subplots(2,2)
    axs[0, 0].imshow(img_bgr)
    axs[0, 1].imshow(markers*128, cmap='gray')
    axs[1, 0].imshow(watshd, cmap='gray')
    plt.show()
    return watshd


def contour_based_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering contours derived from edges."""
    contours = []
    img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3)
    contours, hierarchy = cv2.findContours(img_edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Select only the contour containing the seed point
    for contour in contours:
        region_mask = np.zeros_like(img_gray)
        region_mask = cv2.drawContours(region_mask, [contour], contourIdx=-1, color=255, thickness=-1)  # Draw Interior
        region_mask = cv2.dilate(region_mask, kernel=np.ones((3, 3)))   # Add border too
        if region_mask[seed_pixel]:
            return region_mask

    return np.zeros_like(img_gray)


def intensity(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    img_bgr = img_bgr.astype('float')
    img_gray = (img_bgr[:, :, 0] + img_bgr[:, :, 1] + img_bgr[:, :, 2]) / 3
    return img_gray.astype('uint8')


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    img_bgr = img_bgr.astype('float')
    img_gray = 0.2126*img_bgr[:, :, 2] + 0.7152*img_bgr[:, :, 1] + 0.0722*img_bgr[:, :, 0]
    return img_gray.astype('uint8')


def value(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_hsv[:, :, 2]


def lightness_from_hsl(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    return img_hls[:, :, 1]


def lightness_from_cielab(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    img_hsl = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return img_hsl[:, :, 0]


def process_on_best_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the `best` channel. """
    best_channel = img_bgr[:, :, 1]  # Why 1?
    return cv2.Canny(best_channel, 100, 200)


def process_on_intensity_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the intensity channel. """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return cv2.Canny(img_lab[:, :, 0], 100, 200)


def parallel_channels_then_combine(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on each channel, then combine them. """
    all_edges = np.zeros(shape=img_bgr.shape[:2], dtype=img_bgr.dtype)
    for band in range(img_bgr.shape[-1]):
        all_edges = np.bitwise_or(all_edges, cv2.Canny(img_bgr[:, :, band], 100, 200))
    return all_edges


def process_in_parallel(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to all channels independently. """
    result = np.zeros_like(img)
    for idx in range(img.shape[2]):
        channel = img[:, :, idx]
        channel = cv2.equalizeHist(channel)
        result[:, :, idx] = channel
    return result


def process_intensity_channel_preserve_chroma(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to intensity channel only. """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 0] = cv2.equalizeHist(img_lab[:, :, 0])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


def pansharpen_mean(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the "simple mean" method. """
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    # Apply mean with panchromatic information
    r_out = panchromatic_img//2 + r_in//2
    g_out = panchromatic_img//2 + g_in//2
    b_out = panchromatic_img//2 + b_in//2
    # Return RGB image
    return np.dstack((r_out, g_out, b_out))


def panshapen_Brovey(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the Brovey method. """
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    # Compute normalization factor
    normalization = panchromatic_img / (r_in + g_in + b_in)
    # Apply mean with respect to color-normalization factor
    r_out = r_in * normalization
    g_out = g_in * normalization
    b_out = b_in * normalization
    # Return RGB image
    return np.dstack((r_out, g_out, b_out)).astype('uint8')


def pansharpen_replace_intensity(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image replacing the intensity, and preserving chromatic information. """
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    # Create RGB composition
    img_rgb = np.dstack((r_in, g_in, b_in))
    # Replace lightness with panchromatic information
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_lab[:, :, 0] = panchromatic_img
    # Return RGB image
    return cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
