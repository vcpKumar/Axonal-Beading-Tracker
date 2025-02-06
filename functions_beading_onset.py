#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:00:42 2025

@author: PRETHEESH KUMAR V C
"""

# %% Import Libraries
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from scipy.stats import mode, trim1
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
from scipy.signal import savgol_coeffs

from skimage.restoration import (
    denoise_wavelet,
    estimate_sigma,
)
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import (
    disk,
    skeletonize,
)

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Border, Side
from PyQt5.QtWidgets import QApplication, QFileDialog

import time
import copy
import numpy as np
import cv2 as cv
import pywt
import pandas as pd
import math
from itertools import combinations
import os
import shutil
import yaml

from numba import jit, prange
from numba import (
    njit,
    int16 as i2,
    uint8 as u1,
    uint16 as u2,
    float32 as f4,
    int8 as i1,
)

from itk import (
    Image,
    HessianToObjectnessMeasureImageFilter,
    MultiScaleHessianBasedMeasureImageFilter,
    image_from_array,
    array_from_image,
)
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Title, Div, Spacer
from bokeh.layouts import row, column

# %% Load Configuration Parameters
# Load parameters from the configuration file
configuration = "configuration_beading_onset.yaml"
try:
    with open(configuration, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Error reading parameters file: {e}")
    exit(1)

# Load key parameters from the configuration
padVal = config["padVal"]
templSz = config["templSz"]
templTol = config["templTol"]
cropTol = config["cropTol"]
opDetails = config["opDetails"]
dstBasePath = config["dstBasePath"]

wavelet = config["wavelet"]
wavLev = config["wavLev"]
wavThld = config["wavThld"]
p1 = config["p1"]
p2 = config["p2"]
s1 = config["s1"]
s2 = config["s2"]
s3 = config["s3"]
s4 = config["s4"]
w1 = config["w1"]
w2 = config["w2"]
w3 = config["w3"]
w4 = config["w4"]
w4 = config["w4"]
sclFact = config["sclFact"]

# %% Structuring Elements for Morphological Filtering
kd1 = disk(1)
kd2 = disk(2)
kd3 = disk(3)
kd5 = disk(5)
kd7 = disk(7)
kd20 = disk(20)

# Create a Gaussian kernel for rigorous bead analysis
kernelG = cv.getGaussianKernel(7, sigma=None, ktype=cv.CV_32F)
kernelG = np.outer(kernelG, kernelG)

# %% Rescale Image Utilities


def scaling(array):
    """
    Rescale an array to the range [0, 1].

    Parameters
    ----------
    array : numpy.ndarray
        Input array with positive values.

    Returns
    -------
    numpy.ndarray
        Rescaled array.
    """
    assert (
        np.min(array) >= 0
    ), "Scaling is implemented only for non-negative values."
    dummy = array - np.min(array)
    return dummy / np.max(dummy)


def zero255(array):
    """
    Rescale an array to the range [0, 255] and convert to uint8.

    Parameters
    ----------
    array : numpy.ndarray
        Input array with positive values.

    Returns
    -------
    numpy.ndarray
        Rescaled array in uint8 format.
    """
    return (255 * scaling(array)).astype(np.uint8)


# %% Delete Folder Contents


def deleteFolderContents(path):
    """
    Deletes the contents of the folder at the specified path if it exists.

    Parameters
    ----------
    path : str
        Path to the folder whose contents need to be deleted.

    Returns
    -------
    None
    """
    if not os.path.isdir(path):
        print(f"The specified path {path} is not a directory.")
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Remove a file
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove a directory
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


# %% Select Folder


def selectFolder(prompt):
    """
    Display a GUI prompt to select a folder and return its path.

    Parameters
    ----------
    prompt : str
        Message displayed on the folder selection dialog.

    Returns
    -------
    str or None
        Path of the selected folder, or None if no folder is selected.
    """
    app = QApplication([])  # Initialize the application
    folderPath = QFileDialog.getExistingDirectory(
        None, prompt
    )  # Open folder dialog

    if folderPath:
        print("Selected folder:", folderPath)
        return folderPath
    else:
        print("No folder selected.")
        return None


# %% Iterate Over Images


def iterateOverImages(folderPath):
    """
    Iterate through all image files in the specified folder and return their paths.

    Parameters
    ----------
    folderPath : str
        Path to the folder containing images.

    Returns
    -------
    list
        List of paths to image files with the specified extensions.
    """
    image_extensions = ".tif"  # Add more extensions if needed
    imageList = []

    for root, _, files in os.walk(folderPath):
        files.sort()  # Sort files in alphabetical order
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.lower().endswith(image_extensions):
                imageList.append(file_path)

    return imageList


# %% Click to Get Axon


def onMouseEvent(event, x, y, flags, param):
    """
    Mouse event handler for collecting user-clicked points on an image.

    Parameters
    ----------
    event : int
        Event code (e.g., left-click or right-click).
    x : int
        X-coordinate of the mouse event.
    y : int
        Y-coordinate of the mouse event.
    flags : int
        Additional event flags (not used).
    param : object
        Additional parameters (not used).

    Returns
    -------
    None
    """
    global image1, clicked_points
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv.circle(image1, (x, y), 2, (255, 0, 255), -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        clicked_points.clear()
        image1 = original_image.copy()
    cv.imshow("Image", image1)


def getClick(image):
    """
    Display an image and allow the user to select points using mouse clicks.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in RGB format.

    Returns
    -------
    list
        List of coordinates of clicked points.
    """
    global image1, clicked_points, original_image
    clicked_points = []
    original_image = zero255(image)

    cv.namedWindow("Image", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Image", onMouseEvent)
    image1 = original_image.copy()
    cv.imshow("Image", image1)
    cv.waitKey(0) & 0xFF
    cv.destroyAllWindows()
    return clicked_points


# %% Initialize the axon
def initAxon(points):
    """
    Defines the start and end points of an axon based on user mouse clicks.
    Computes the slopes at the start and end points and adjusts coordinates
    by adding a padding value for template matching.

    ---------------------------PARAMETERS--------------------------------------

    points : List of tuples
        A list where each tuple contains two integers representing the
        coordinates of mouse clicks on the image.
    padVal : int
        Padding size (equal on all sides), used for template matching.

    -----------------------------RETURNS---------------------------------------

    sc : int
        Column coordinate of the start of the axon.
    sr : int
        Row coordinate of the start of the axon.
    ec : int
        Column coordinate of the end of the axon.
    er : int
        Row coordinate of the end of the axon.
    slope_s : float
        Slope at the start of the axon.
    slope_e : float
        Slope at the end of the axon.
    orient : str
        Orientation of the axon: either 'vert' (vertical) or 'horz' (horizontal).

    """
    p_e1 = points[0]
    p_e2 = points[-1]  # Two end points of the axon

    # Determine orientation based on the difference between x and y coordinates
    if abs(p_e1[1] - p_e2[1]) > abs(p_e1[0] - p_e2[0]):
        orient = "vert"
        if p_e1[1] < p_e2[1]:
            p_s, p_e = p_e1, p_e2  # Set p_s as start and p_e as end
        else:
            p_s, p_e = p_e2, p_e1
    else:
        orient = "horz"
        if p_e1[0] < p_e2[0]:
            p_s, p_e = p_e1, p_e2
        else:
            p_s, p_e = p_e2, p_e1

    # Calculate slopes based on orientation
    if orient == "horz":
        slope_s = (p_e1[0] - p_e2[0]) / (p_e1[1] - p_e2[1])  # Horizontal slope
        slope_e = copy.deepcopy(slope_s)
    else:  # Vertical orientation
        slope_s = (p_e1[1] - p_e2[1]) / (p_e1[0] - p_e2[0])  # Vertical slope
        slope_e = copy.deepcopy(slope_s)

    # Adjust coordinates with padding for template matching
    sr = p_s[1] + padVal
    sc = p_s[0] + padVal
    er = p_e[1] + padVal
    ec = p_e[0] + padVal

    return np.array((sc, sr, ec, er), np.int16), slope_s, slope_e, orient


# %% TEMPLATE MATCHING-MULTI ANGLE


def templateMatch_angled(image, template):
    """
    Perform template matching for determining the terminations of the axon.
    Matching is performed by rotating the template at specified angles.

    ---------------------------- Parameters ------------------------------

    image : numpy.ndarray, uint8
        The image in which to search.
    template : numpy.ndarray, uint8
        The template to match.

    ----------------------------- Returns -------------------------------

    best_match: tuple
        The best match location in (x, y) format.
        (x, y) is the position in the image where the top-left corner of the
        template is placed for the best match.
    """
    angles = range(-6, 7, 2)  # Angles for rotation (-6 to +6 degrees, step=2)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(rotateNmatch, image, template, angle)
            for angle in angles
        ]
        bestMatch = None
        bestVal = -np.inf

        for future in as_completed(futures):
            corrVal, corrLoc = future.result()
            if corrVal > bestVal:
                bestVal = corrVal
                bestMatch = corrLoc

    return bestMatch  # Return best match coordinates


def rotateNmatch(image, template, angle):
    """
    Helper function for 'templateMatch_angled'.
    Rotates the template by a given angle and performs template matching.

    ----------------------- Parameters ---------------------------------

    image : numpy.ndarray, uint8
        The image in which to search.
    template : numpy.ndarray, uint8
        The template to match.
    angle : int
        The angle to rotate the template.

    -------------------------- Returns ---------------------------------

    tuple
        The maximum value and location of the best match.
    """
    w, h = template.shape
    M = cv.getRotationMatrix2D((templSz, templSz), angle, 1)
    rotated_template = cv.warpAffine(template, M, (w, h))
    rotated_template = rotated_template[templTol:-templTol, templTol:-templTol]
    result = cv.matchTemplate(image, rotated_template, cv.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    return maxVal, maxLoc


# %% Matched Coordinates
def getMatchedCords(c1, r1, best_match, h, w):
    """
    Convert the location of the best match to actual image coordinates.

    ---------------------------- Parameters -----------------------------

    r1 : int
        Row offset for the cropped image.
    c1 : int
        Column offset for the cropped image.
    best_match : tuple
        Coordinates of the best match within the cropped image.
    h : int
        Height of the original image.
    w : int
        Width of the original image.

    ---------------------------- Returns -------------------------------

    numpy.ndarray
        The converted row and column coordinates of the best match.
    """
    r = r1 - padVal + best_match[1] + templSz - templTol
    c = c1 - padVal + best_match[0] + templSz - templTol
    r = max(r, padVal)
    r = min(r, h + padVal)
    c = max(c, padVal)
    c = min(c, w + padVal)
    return np.array((c, r), np.int16)


# %% CROPPING
def cropping(im, mask, cropBox, h, w):
    """
    Crop the region of interest based on the location of the axon.

    ----------------------------- Parameters ---------------------------

    im : numpy.ndarray, uint16
        The grayscale image to be cropped.
    mask : numpy.ndarray, uint8
        The mask from the previous iteration.
    cropBox : list of int
        The [min_row, max_row, min_col, max_col] defining the cropping box.
    cropTol : int
        The tolerance to be added to the cropping box.
    h : int
        Height of the image.
    w : int
        Width of the image.

    ----------------------------- Returns ------------------------------

    tuple
        Cropped grayscale image, color image, mask, shape, and adjusted crop box.
    """
    yCrop1 = max(cropBox[0] - cropTol, 0)
    yCrop2 = min(cropBox[1] + cropTol, h)
    xCrop1 = max(cropBox[2] - cropTol, 0)
    xCrop2 = min(cropBox[3] + cropTol, w)

    imCrop = im[yCrop1:yCrop2, xCrop1:xCrop2]
    maskCrop = mask[yCrop1:yCrop2, xCrop1:xCrop2].astype(np.uint8) * 255
    imColrCrop = cv.cvtColor(zero255(imCrop), cv.COLOR_GRAY2BGR)
    cropShp = np.shape(imCrop)
    cropBox = [yCrop1, yCrop2, xCrop1, xCrop2]

    return imCrop,  imColrCrop, maskCrop, cropShp, cropBox


# %% Meijering Filter
def itkMeijering(im):
    """
    Apply the Meijering filter for neuron or vessel enhancement.

    ----------------------------- Parameters ---------------------------

    im : itk.itkImagePython.itkImageSS2
        The image for neuron enhancement, converted to an ITK-compatible format.

    ----------------------------- Returns ------------------------------

    numpy.ndarray
        Image with enhanced neuron structures.
    """
    meijering_measure = HessianToObjectnessMeasureImageFilter.New()
    meijering_measure.SetObjectDimension(1)  # Line-like structures
    meijering_measure.SetBrightObject(False)  # Detect dark structures

    meijering_measure.SetAlpha(0.3)  # Sensitivity parameter

    multi_scale_filter = MultiScaleHessianBasedMeasureImageFilter.New()
    multi_scale_filter.SetInput(im)
    multi_scale_filter.SetSigmaMinimum(2.0)
    multi_scale_filter.SetSigmaMaximum(7.0)
    multi_scale_filter.SetNumberOfSigmaSteps(2)
    multi_scale_filter.SetHessianToMeasureFilter(meijering_measure)

    multi_scale_filter.Update()
    opIm = multi_scale_filter.GetOutput()

    return array_from_image(opIm)


# %% CONNECTED COMPONENT ANALYSIS
def binaryFilter(imBin, shape, n):
    """
    Filters out small connected components from a binary image that do not
    meet the minimum pixel threshold.

    Parameters
    ----------
    imBin : numpy.ndarray, uint8
        The binary image to be filtered.
    shape : tuple
        The shape of the binary image.
    n : int
        Minimum pixel count for connected components to be retained.

    Returns
    -------
    imBinFilt : numpy.ndarray, uint8
        The filtered binary image.
    """
    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(
        imBin, cv.CV_32S
    )
    imBinFilt = np.zeros(shape, dtype="uint8")
    for i in range(1, numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > n:
            componentMask = (labels == i).astype("uint8") * 255
            imBinFilt = cv.bitwise_or(imBinFilt, componentMask)
    return imBinFilt


# %% CURVE FITTING
def fitCurve(skltn, x0, y0, xn, yn, orient, initGuess, maxfev):
    """
    Perform curve fitting on the skeletonized axon to represent its shape
    using a polynomial model.

    Parameters
    ----------
    skltn : numpy.ndarray
        The skeletonized version of the axon.
    x0 : int
        Starting x-coordinate of the axon.
    y0 : int
        Starting y-coordinate of the axon.
    xn : int
        Ending x-coordinate of the axon.
    yn : int
        Ending y-coordinate of the axon.
    orient : str
        Orientation of the axon ('vert' or 'horz').
    initGuess : numpy.ndarray
        Initial values of the polynomial coefficients.
    maxfev : int
        Maximum number of iterations for curve fitting.

    Returns
    -------
    xVect : numpy.ndarray, dtype=int
        X-coordinates along the axon for the fitted curve.
    yVect : numpy.ndarray, dtype=int
        Y-coordinates along the axon for the fitted curve.
    slopes : numpy.ndarray, dtype=float
        Slopes along the axon for the fitted curve.
    params : numpy.ndarray, dtype=float32
        Polynomial coefficients after fitting.
    """
    if orient == "horz":
        xVect, yVect, slopes, params = curveFitting(
            skltn, x0, xn, orient, initGuess, maxfev
        )
    else:
        yVect, xVect, slopes, params = curveFitting(
            skltn, y0, yn, orient, initGuess, maxfev
        )

    return xVect, yVect, np.float32(slopes), params


def curveFitting(skltn, indVect_0, indVect_n, orient, initGuess, maxfev):
    """
    Helper function for `fitCurve`. Performs polynomial fitting on the axon
    skeleton.

    Parameters
    ----------
    skltn : numpy.ndarray
        The skeletonized version of the axon.
    indVect_0 : int
        Starting value of the independent coordinate.
    indVect_n : int
        Ending value of the independent coordinate.
    orient : str
        Orientation of the axon ('vert' or 'horz').
    initGuess : numpy.ndarray
        Initial values of the polynomial coefficients.
    maxfev : int
        Maximum number of iterations for curve fitting.

    Returns
    -------
    indVect : numpy.ndarray, dtype=int
        Independent coordinates along the axon for the fitted curve.
    dependVect : numpy.ndarray, dtype=int
        Dependent coordinates along the axon for the fitted curve.
    slopes : numpy.ndarray, dtype=float
        Slopes along the axon for the fitted curve.
    params : numpy.ndarray, dtype=float32
        Polynomial coefficients after fitting.
    """
    y, x = np.where(skltn)
    if orient == "horz":
        params, cov = curve_fit(
            polynomial, x, y, p0=initGuess, maxfev=maxfev, xtol=1e-6, gtol=1e-6
        )
    else:
        params, cov = curve_fit(
            polynomial, y, x, p0=initGuess, maxfev=maxfev, xtol=1e-6, gtol=1e-6
        )

    poly = np.poly1d(params)
    indVect = np.linspace(
        indVect_0, indVect_n, indVect_n - indVect_0 + 1
    ).astype(np.int16)
    dependVect = np.round(poly(indVect)).astype(np.int16)
    d_poly = poly.deriv()
    slopes = (d_poly(indVect)).astype(np.float32)

    return indVect, dependVect, slopes, params


def polynomial(x, *coeffs):
    """
    Polynomial function for curve fitting.

    Parameters
    ----------
    x : float
        Input value.
    coeffs : tuple
        Polynomial coefficients.

    Returns
    -------
    float
        Computed polynomial value.
    """
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))


# %% Get perpendicular end coordinates
@njit(
    "Tuple((i2[:], i2[:], i2[:], i2[:]))(i2[:], i2[:], f4[:], i1, u1)",
    nogil=True,
    cache=True,
)
def getPerpendCords(xVect, yVect, slopes, orientFlag, L):
    """
    Calculate the coordinates of perpendicular lines for given points and slopes.

    Parameters
    ----------
    xVect : numpy.ndarray
        x-coordinates of the axon spine.
    yVect : numpy.ndarray
        y-coordinates of the axon spine.
    slopes : numpy.ndarray
        Slopes at each point.
    orientFlag : int
        Orientation flag (1 for horizontal, 0 for vertical).
    L : int
        Length of the perpendicular lines.

    Returns
    -------
    xs, ys : numpy.ndarray
        x and y coordinates of the starting points of the perpendicular lines.
    xe, ye : numpy.ndarray
        x and y coordinates of the ending points of the perpendicular lines.
    """
    # Compute unit vectors for the perpendicular slopes
    if orientFlag == 1:  # Horizontal orientation
        slopes_p = -1 / slopes  # Perpendicular slopes in Cartesian coordinates
    else:  # Vertical orientation
        slopes_p = -1 * slopes

    # Create direction vectors for perpendicular slopes
    d_vectors = np.empty((2, slopes.shape[0]), dtype=np.float32)
    d_vectors[0, :] = 1.0  # Fill the x-component with ones
    d_vectors[1, :] = slopes_p
    norms = np.sqrt(d_vectors[1, :] ** 2 + d_vectors[0, :] ** 2)
    u_vectors = d_vectors / norms  # Normalize the direction vectors

    # Allocate arrays for the coordinates of the perpendicular lines
    xs = np.empty_like(xVect, dtype=np.int16)
    ys = np.empty_like(yVect, dtype=np.int16)
    xe = np.empty_like(xVect, dtype=np.int16)
    ye = np.empty_like(yVect, dtype=np.int16)

    # Calculate the start and end points of the perpendicular lines
    xs[:] = xVect + L * u_vectors[0, :]  # Start points (x)
    ys[:] = yVect + L * u_vectors[1, :]  # Start points (y)
    xe[:] = xVect - L * u_vectors[0, :]  # End points (x)
    ye[:] = yVect - L * u_vectors[1, :]  # End points (y)

    return xs, ys, xe, ye


# %% Bresenham's Algorithm
@njit("i2[:, :](i2, i2, i2, i2)", nogil=True, cache=True)
def bresenhamAlg(x0, y0, x1, y1):
    """
    Generate points of a line using Bresenham's algorithm optimized with Numba.

    Parameters
    ----------
    x0, y0 : int
        Starting coordinates of the line.
    x1, y1 : int
        Ending coordinates of the line.

    Returns
    -------
    np.ndarray
        A 2D array where each row represents a point (x, y) on the line.
    """
    # Estimate the maximum number of points required (length of the diagonal)
    max_length = int(np.hypot(x1 - x0, y1 - y0)) + 1
    points = np.empty((max_length, 2), dtype=np.int16)
    point_count = 0

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1  # Step for x direction
    sy = 1 if y0 < y1 else -1  # Step for y direction
    err = dx + dy  # Initial error term

    while True:
        # Store the current point
        points[point_count, 0] = x0
        points[point_count, 1] = y0
        point_count += 1

        # Check if the line endpoint is reached
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    # Return only the valid points
    return points[:point_count]  # Output is in (x, y) format


# %% Clubbing of points
@njit("f4[:, :](f4[:, :])", nogil=True, cache=True)  # Numba Decorator
def clubbing(points):
    """
    Helper function for calcAxonDia.
    Groups pixels that are less than 1.42 units apart into one.

    Parameters
    ----------
    points : numpy.ndarray, float32
        Input array containing point coordinates.

    Returns
    -------
    output : numpy.ndarray, float32
        Output array with points grouped if the conditions are met.

    """
    # Preallocate arrays for grouping points into club1 and club2
    N = len(points)
    club1 = np.empty((N, 2), dtype=np.float32)
    club2 = np.empty((N, 2), dtype=np.float32)
    c1Count = 0
    c2Count = 0
    # Mask to mark grouped points
    grouped = np.zeros(N, dtype=np.bool_)

    # Iterate through all possible pairs of points
    for i in range(N):
        # Start comparisons from the next point
        for j in range(i + 1, N):
            # Select two points
            p1 = points[i]
            p2 = points[j]
            diff = p1 - p2
            distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

            # If the distance between p1 and p2 is less than 1.42
            if distance < 1.42:

                # Case 1: If club1 is empty, place both p1 and p2 there
                if c1Count == 0:
                    club1[c1Count] = p1
                    club1[c1Count + 1] = p2
                    c1Count += 2
                    grouped[i] = True
                    grouped[j] = True
                    break

                # Case 2: If p1 is in club1, add p2 to it
                p1_inClub1 = False
                for k in range(c1Count):
                    if (p1 == club1[k]).all():
                        p1_inClub1 = True
                        break
                if p1_inClub1:
                    club1[c1Count] = p2
                    c1Count += 1
                    grouped[j] = True
                    break

                # Case 3: If p2 is in club1, add p1 to it
                p2_inClub1 = False
                for k in range(c1Count):
                    if (p2 == club1[k]).all():
                        p2_inClub1 = True
                        break
                if p2_inClub1:
                    club1[c1Count] = p1
                    c1Count += 1
                    grouped[i] = True
                    break

                # Case 4: If club2 is empty, place both p1 and p2 there
                if c2Count == 0:
                    club2[c2Count] = p1
                    club2[c2Count + 1] = p2
                    c2Count += 2
                    grouped[i] = True
                    grouped[j] = True
                    break

                # Case 5: If p1 is in club2, add p2 to it
                p1_inClub2 = False
                for k in range(c2Count):
                    if (p1 == club2[k]).all():
                        p1_inClub2 = True
                        break
                if p1_inClub2:
                    club2[c2Count] = p2
                    c2Count += 1
                    grouped[j] = True
                    break

                # Case 6: If p2 is in club2, add p1 to it
                p2_inClub2 = False
                for k in range(c2Count):
                    if (p2 == club2[k]).all():
                        p2_inClub2 = True
                        break
                if p2_inClub2:
                    club2[c2Count] = p2
                    c2Count += 1
                    grouped[i] = True
                    break

    # Calculate the number of ungrouped points
    ungrouped = N - np.sum(grouped)
    nFinal = ungrouped
    if c1Count != 0:
        nFinal += 1
    if c2Count != 0:
        nFinal += 1
    if nFinal != 2:
        return np.empty((0, 2), dtype=np.float32)

    # Combine the results
    output = np.empty((2, 2), dtype=np.float32)
    dummy = 0

    # Add the mean of club1 to the output
    if c1Count > 0:
        club1_mean = np.sum(club1[:c1Count], axis=0) / c1Count
        output[dummy, :] = club1_mean
        dummy += 1

    # Add the mean of club2 to the output
    if c2Count > 0:
        club2_mean = np.sum(club2[:c2Count], axis=0) / c2Count
        output[dummy, :] = club2_mean
        dummy += 1

    # Add ungrouped points to the output
    for i in range(N):
        if not grouped[i]:
            output[dummy, :] = points[i]
            dummy += 1

    return output


# %% Edge cuts detection
@njit("f4[:,:](u1[:, :], i2[:, :])", nogil=True, parallel=True, cache=True)
def findEdgeCuts(edges, line_points):
    """
    Helper function for calcAxonDia.
    Finds the intersection points between line_points and edges.

    Parameters
    ----------
    edges : numpy.ndarray, uint8
        Binary image representing the axon boundaries.
    line_points : numpy.ndarray, int16
        Pixel coordinates along a perpendicular line to the axon.

    Returns
    -------
    edge_cuts : numpy.ndarray, float32
        Coordinates of the intersection points between the perpendicular line and the axon boundary.

    """
    # Create a mask to mark valid points
    mask = np.zeros(len(line_points), dtype=np.int32)
    for idx in prange(len(line_points)):  # Parallelized loop
        x, y = line_points[idx]
        if edges[y, x] != 0:
            mask[idx] = 1  # Mark as a valid point

    # Count valid points and allocate the output array
    total_valid = np.sum(mask)
    edge_cuts = np.empty((total_valid, 2), dtype=np.float32)

    # Populate the output array with valid points
    pos = 0
    for idx in range(len(line_points)):
        if mask[idx] == 1:
            edge_cuts[pos] = line_points[idx]
            pos += 1

    return edge_cuts


# %% Calculate axon diameter
@njit(
    "f4[:](i2[:], i2[:], i2[:], i2[:], u1[:, :], i2[:], i2[:])",
    nogil=True,
    parallel=True,
    cache=True,
)
def calcAxonDia(xs, ys, xe, ye, edges, y_vect, x_vect):
    """
    Calculates the axon diameter at each point along the spine (medial axis).
    At each pixel on the spine, a perpendicular line is specified by two points (xs, ys) and (xe, ye).

    Parameters
    ----------
    xs : numpy.ndarray, int16
        x-coordinates of the start points of the perpendicular lines.
    ys : numpy.ndarray, int16
        y-coordinates of the start points of the perpendicular lines.
    xe : numpy.ndarray, int16
        x-coordinates of the end points of the perpendicular lines.
    ye : numpy.ndarray, int16
        y-coordinates of the end points of the perpendicular lines.
    edges : numpy.ndarray, uint8
        Binary image containing the edges of the axon.
    y_vect : numpy.ndarray, int16
        y-coordinates of the spine.
    x_vect : numpy.ndarray, int16
        x-coordinates of the spine.

    Returns
    -------
    dias : numpy.ndarray, float32
        Array containing the diameters along the axon.

    """
    # Length of the spine in pixels
    n = len(xs)

    # Initialize the diameters array
    dias = np.empty(n, dtype=np.float32)

    # Iterate over the length of the axon
    for i in prange(n):
        # Define the perpendicular line's start and end points
        x1, y1, x2, y2 = xs[i], ys[i], xe[i], ye[i]

        # Get pixel coordinates along the perpendicular line
        line_points = bresenhamAlg(x1, y1, x2, y2)

        # Find the intersections between the perpendicular line and the axon edges
        edge_cuts = findEdgeCuts(edges, line_points)

        # If more than one intersection is found, group close points and calculate diameter
        if edge_cuts.shape[0] > 1:
            edge_cuts = clubbing(edge_cuts)
            if edge_cuts.shape[0] == 2:
                diff = edge_cuts[0] - edge_cuts[1]
                dias[i] = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            else:
                dias[i] = 0.0
        else:
            dias[i] = 0.0

    return dias


# %% 1-D WAVELET DENOISING
def denoiseWavelet_1D(data, wavelet, thldFactor, level=3):
    """
    wavelet denoising for 1-D data

    Parameters
    ----------
    data : numpy.ndarray,float32
        the diameter along the axon
    wavelet : str
        the wavelet to be used for denoising, to be selected the ones available
        in pywavelets
    thldFactor : float
        affects the thresholding of wavelet coefficients
    level : int, optional
        The number of levels to be used The default is 3.

    Returns
    -------
    denoised_data : numpy.ndarray,float32
        the diameter values after denoising

    """
    coeffs = pywt.wavedec(data, wavelet, level)
    cD1 = coeffs[-1]
    sigma_est = np.median(np.abs(cD1 - np.median(cD1))) / 0.6745
    thld = sigma_est * np.sqrt(2 * np.log(len(data))) * thldFactor
    denoised_coeffs = [pywt.threshold(c, thld, mode="soft") for c in coeffs]
    denoised_data = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_data


# %% SAvgol Filter
def savgolFilt(y, window_length, polyorder):
    """
    Savgol Filter for smoothing 1-D data. This is used in the rigorous anlysis
    beading, where the number of data points are quite small.
    Parameters
    ----------
    y : numpy.ndarray,float32
       the diameter values of the cropped bead
    window_length : int
        window length for filtering
    polyorder : int
        polynomial order to be used

    Returns
    -------
    y_smooth :numpy.ndarray,float32
        filtered data
    y_deriv : numpy.ndarray,float32
        the first derivative of y_smooth. high values of derivative implies
        that the peak is arising from some debries or protrusions

    """
    coeffs = savgol_coeffs(window_length, polyorder)

    # Pad the data to handle boundaries
    pad_width = window_length // 2
    y_padded = np.pad(y, pad_width, mode="wrap")

    # Apply the filter to get smoothed data
    y_smooth = np.convolve(y_padded, coeffs, mode="valid")

    # Calculate Savitzky-Golay coefficients for the first derivative
    deriv_coeffs = savgol_coeffs(window_length, polyorder, deriv=1)

    # Apply the filter to get the first derivative
    y_deriv = np.convolve(y_padded, deriv_coeffs, mode="valid")

    return y_smooth, y_deriv


#
# %% Zero Crossings


def findZeroCrossings(laplcn):
    """
    Finds the zero Crossings in the lapalcian

    Parameters
    ----------
    laplcn : array of float


    Returns
    -------
    zeroCrossings : array of uint8
        A bianry image with 1 at pixels with a zero crossing

    """

    r, c = np.shape(laplcn)
    zeroCrossings = np.zeros((r, c), dtype=np.uint8)  # initialize the array

    # create two copies of the Lapalcian
    laplcnPostv = np.copy(laplcn)
    laplcnNegtv = np.copy(laplcn)
    # keep one copy with positve values and vice versa
    laplcnPostv[laplcnPostv < 0] = 0  # Keep only positive values
    laplcnNegtv[laplcnNegtv > 0] = 0  # Keep only negative values
    laplcnNegtv = np.abs(laplcnNegtv)  # magnitude ofnegative values are taken
    # %% Find the zero crossing

    # the strength for a valid zero-crossing
    thld = 0.04 * np.max(laplcnNegtv + laplcnPostv)

    # Convolve both regions with the kernel
    # For a given pixel the sum of positive second Derivatives around its
    # neighborhood (decided by the kernel)is found
    convPostv = cv.filter2D(laplcnPostv, ddepth=-1, kernel=kernelG)
    # same for negative second derivatives (magnitude)
    convNegtv = cv.filter2D(laplcnNegtv, ddepth=-1, kernel=kernelG)
    # Calculate combined response
    combined = convPostv + convNegtv
    max_response = np.maximum(convPostv, convNegtv)
    # Identify zero crossings (vectorized)
    # One more condition is added to make sure contribution from either sides
    zeroCrossings = (combined > thld) & (combined > 1.2 * max_response)
    return zeroCrossings


# %% The Rigorous bead analysis
def rigBeadAnal(
    beadCrop,
    beadBin,
    dia,
    orient,
    initGuess1,
    file,
    k,
    i,
    dstPath,
):
    """
    Perform rigorous analysis to confirm the presence of a bead.

    Parameters
    ----------
    beadCrop : numpy.ndarray, uint16
        Cropped image segment containing a possible bead location.
    beadBin : numpy.ndarray, uint8
        Corresponding binary image of the cropped region.
    dia : float
        Approximate diameter of the bead in pixels.
    orient : str
        Orientation of the bead, either 'vert' (vertical) or 'horz' (horizontal).
    initGuess1 : float32
        Initial guess for curve-fitting parameters.
    file : str or file-like object
        File used to store analysis results.
    k : int
        Index of the current image in the series, used for saving results.
    i : int
        Index of the current possible bead being analyzed within the image.
    dstPath : str
        Path for saving the results.

    Returns
    -------
    bool
        True if the analyzed region contains a bead, False otherwise.

    """
    # Check if the binary region is sufficiently large to analyze
    if np.count_nonzero(beadBin) < 100:
        print("\tNumber of non-zero points is less than 100", file=file)
        return False

    # Preprocess the inputs
    r, c = np.shape(beadCrop)

    # Filter out small isolated regions in the binary image
    beadBin = binaryFilter(beadBin, (r, c), 100)

    # Remove very thick regions to facilitate fitting along the medial axis
    kernelB = disk(int(dia / 2 + 3))
    beadBin1 = cv.morphologyEx(beadBin, cv.MORPH_TOPHAT, kernelB)

    # Smooth the binary image after removing thick regions
    beadBin1 = cv.erode(beadBin1, kd2)

    # Skeletonize the binary image to obtain the medial axis
    beadSkltn = skeletonize(beadBin1)

    # Perform curve fitting to extract the medial axis of the bead
    xFit1, yFit1, slpsB, _ = fitCurve(
        beadSkltn, 7, 7, c - 7, r - 7, orient, initGuess1, 50
    )

    # %% Create the masks
    # Isolate the central blob
    centreBlob = beadBin - beadBin1
    centreBlob = cv.erode(centreBlob, kd7)  # Erode to refine the blob
    centreBlob = cv.erode(centreBlob, kd1, iterations=2)
    centreBlob = 1 - centreBlob / 255  # Create a mask from the central blob

    # Create another mask to isolate the edges of interest
    beadMask = cv.dilate(beadBin, kd2, iterations=2)

    # %% Find the contours of the bead

    # Apply Gaussian smoothing to the cropped bead region
    beadCrop = cv.GaussianBlur(beadCrop, (17, 17), sclFact)

    # Apply the Laplacian filter to detect edges
    laplcn = cv.Laplacian(beadCrop, cv.CV_32F, ksize=7)

    # Find zero-crossings in the Laplacian for edge detection
    edgesB = findZeroCrossings(laplcn).astype(np.uint8)

    # Suppress unwanted edges by applying masks
    edgesB = (edgesB * beadMask * centreBlob).astype(np.uint8)

    # Filter out small isolated edges
    edgesB = binaryFilter(edgesB, (r, c), 20)

    # %% Save the edges and mask if needed
    # This is controlled by the configuration flag `opDetails`
    if opDetails:
        # Combine and save the mask and edges for debugging or analysis
        beadStruct = np.dstack(
            (
                beadMask / 2,
                (1 - centreBlob) * 200,
                edgesB + (1 - centreBlob) * 200,
            )
        )
        cv.imwrite(f"{dstPath}/{k}_{i}_beadAnal.png", beadStruct)
        cv.imwrite(f"{dstPath}/{k}_{i}_bead.png", beadCrop)

    # %% Find the diameter along the beaded part of the axon

    # Pad the edges to avoid boundary issues during diameter calculation
    p = 10
    edgesP = np.pad(edgesB, pad_width=p, mode="constant", constant_values=0)

    # Update dimensions and coordinates after padding
    r1, c1 = np.shape(edgesP)
    xFit1 += p
    yFit1 += p

    # Adjust orientation flag for compatibility with Numba
    orientFlag = np.int8(1) if orient == "horz" else np.int8(-1)

    # Get perpendicular points on either side of the medial axis
    xs, ys, xe, ye = getPerpendCords(
        xFit1, yFit1, slpsB, orientFlag, L=np.uint8(dia + 5)
    )

    # Calculate the diameter along the edges
    diasB = calcAxonDia(xs, ys, xe, ye, edgesP, yFit1, xFit1)

    # %% Perform calculations on the diameter
    # Replace missing diameter values (0) with NaN
    diasB[diasB == 0] = np.nan

    # Interpolate to fill in missing points, allowing gaps up to 5
    diasB = pd.Series(diasB).interpolate(limit=5, limit_direction="forward")

    # Smooth the diameter values using a Savitzky-Golay filter
    diasB = savgol_filter(diasB, 11, polyorder=3)

    # Calculate the gradient of the smoothed diameter
    grads = np.abs(np.gradient(diasB))
    # %% Find peaks in the diameter profile
    pks, prop = find_peaks(
        diasB,
        height=dia * 2 / 3,  # Minimum height of peaks
        threshold=None,  # No specific threshold for neighboring points
        distance=50,  # Minimum distance between peaks
        prominence=tanhDecd(
            dia, lwr=p1, upr=p2
        ),  # Prominence threshold (Tp in paper)
        width=(1, 50),  # Allowable peak widths
        rel_height=0.5,  # Relative height for width calculation
    )

    # %% If no peaks are detected or criteria are violated
    cnfmB = False  # Flag indicating whether a bead is confirmed
    propsB = []  # To store bead properties
    if len(pks) == 0:
        print(
            "\t \t No valid peak detected or multiple peaks with sufficient prominence.",
            file=file,
        )
        return propsB, cnfmB

    # %% Extract properties of the detected peak

    width = np.round(prop["widths"][0], 2)  # Full-width at half-maximum (FWHM)
    prom = np.round(prop["prominences"][0], 2)  # Prominence of the peak
    widthMax = (prop["right_bases"] - prop["left_bases"])[
        0
    ]  # Base width of the peak

    # Extract gradient values near the peak base
    lBase = prop["left_bases"][0] + 2  # Left base with padding
    rBase = prop["right_bases"][0] - 2  # Right base with padding
    grads1 = grads[lBase:rBase]  # Gradients in the region

    # Check for high slope instances in the vicinity of the peak
    highSlpInstB = np.sum(
        grads1 > tanhDecd(dia, s3, s4)
    )  # Count of high slopes (Ts' in paper)
    highSlpB = np.round(np.max(grads1), 2)  # Maximum gradient

    # %% Save the extracted properties
    propsB.append((highSlpB, highSlpInstB))  # High slope properties
    propsB.append(prom)  # Peak prominence
    propsB.append(width)  # Peak FWHM
    propsB.append(widthMax)  # Peak base width

    # %% Check various criteria to confirm bead detection

    # Slope criteria: Too many high slopes in the region
    if highSlpInstB > 1:
        print(
            "\t \t Too many high slopes in bead analysis:",
            highSlpInstB,
            file=file,
        )
        return propsB, cnfmB

    # Flat, wide structures with a slight bump should not be detected as beads
    if 8 * prom < widthMax:
        print(
            "\t \t Prominence-to-width error detected.",
            file=file,
        )
        return propsB, cnfmB

    # %% Additional criteria for regions with high slopes

    # Define slope threshold (Ts in paper)
    Ts = tanhDecd(dia, s1, s2)

    # Tw criterion: Compare width and prominence
    if highSlpB > Ts and widthMax < tanhDecd(dia, w1, w2) * width:
        print(
            "\t \t WidthMax comparison failed.",
            widthMax,
            width,
            file=file,
        )
        return propsB, cnfmB

    # Tw' criterion: Compare width and base width with prominence
    if highSlpB > Ts and tanhDecd(dia, w3, w4) * prom > width:
        print(
            "\t \t Width-prominence comparison failed.",
            width,
            prom,
            file=file,
        )
        return propsB, cnfmB

    # %% Bead confirmed if all criteria are met
    else:
        print(
            "\t \t Bead confirmed with slope=",
            format(highSlpB, ".2f"),
            file=file,
        )
        cnfmB = True  # Mark as a confirmed bead

        # Save details if the configuration flag `opDetails` is not enabled
        if not opDetails:
            centreBlob[:7, :] = centreBlob[-7:, :] = centreBlob[:, :7] = (
                centreBlob[:, -7:]
            ) = 0  # Clear edges
            beadStruct = np.dstack(
                (beadMask / 2, (1 - centreBlob) * 200, edgesB)
            )
            cv.imwrite(
                f"{dstPath}/{k}_{i}_beadAnal.png",
                beadStruct,
            )

        # Return bead properties and confirmation flag
        return propsB, cnfmB


# %% Tanh-based decision function
def tanhDecd(dia, lwr, upr, diaM=15.5, shrpns=1):
    """
    Compute a value based on a hyperbolic tangent (tanh) function,
    used for thresholds that vary smoothly with the diameter.

    Parameters
    ----------
    dia : float
        Diameter value for which the decision is being computed.
    lwr : float
        Lower bound of the decision threshold.
    upr : float
        Upper bound of the decision threshold.
    diaM : float, optional
        Midpoint diameter value where the threshold transitions (default is 15.5).
    shrpns : float, optional
        Sharpness of the transition in the threshold (default is 1).

    Returns
    -------
    float16
        Computed threshold value based on the tanh function.
    """
    return (
        lwr + (upr - lwr) / 2 * (1 + np.tanh(shrpns * (dia - diaM)))
    ).astype(np.float16)


# %% Convert image to Bokeh format
def image2BokehFormat(image):
    """
    Convert an image to a format compatible with Bokeh visualization.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array. Can be RGB or RGBA.

    Returns
    -------
    numpy.ndarray
        Converted image in a 32-bit RGBA format suitable for Bokeh.

    Raises
    ------
    ValueError
        If the input image format is not supported.

    Notes
    -----
    - If the input image has 3 channels (RGB), an alpha channel is added with a value of 255.
    - If the input image already has 4 channels (RGBA), the alpha channel is preserved.
    """
    if image.ndim == 3:  # Check if the image has multiple channels
        # Initialize an empty 32-bit RGBA image
        rgba_image = np.empty(
            (image.shape[0], image.shape[1]), dtype=np.uint32
        )
        # Create a view of the RGBA image as an 8-bit array
        view = rgba_image.view(dtype=np.uint8).reshape(
            (image.shape[0], image.shape[1], 4)
        )
        # Copy RGB channels from the input image
        view[:, :, 0:3] = image[:, :, 0:3]
        # Handle the alpha channel
        if image.shape[2] == 3:  # If only RGB, set alpha to 255
            view[:, :, 3] = 255
        else:  # If RGBA, copy the alpha channel
            view[:, :, 3] = image[:, :, 3]
        return rgba_image
    else:
        raise ValueError(
            "Unsupported image format. Only RGB and RGBA are supported."
        )
