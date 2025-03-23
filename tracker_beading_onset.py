#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:02:06 2025

@author: PRETHEESH KUMAR V C
"""

from functions_beading_onset import *

# %% Start the program
# Prompt the user to select a folder containing the image series

# srcPath = selectFolder("Select the folder with the image series")
# if srcPath is None:
#     print("No folder selected. Exiting.")
#     sys.exit(1)
srcPath = "100_nM_Vincristine"
# Get file IDs by iterating over images in the selected folder
fileIds = iterateOverImages(srcPath)

# Prompt the user to select the output folder
# Output folder path is initialized
dstPath = dstBasePath
k_in = 0

# Ensure the base output directory does not already exist
# Ensure the base output directory does not already exist
counter = 2
while os.path.exists(dstPath):
    dstPath = os.path.join(os.path.dirname(dstBasePath),
                           f"{os.path.basename(dstBasePath)}{counter:02d}")
    counter += 1

# Create the output directory
os.makedirs(dstPath)

# Clear the contents of the output folder if it exists
deleteFolderContents(dstPath)

# %% Open files to store results
file1 = open(
    os.path.join(dstPath, "Rig_Bead_Anals.txt"), "w"
)  # File for bead analysis results
file2 = open(
    os.path.join(dstPath, "Bead_Cords.txt"), "w"
)  # File for storing bead coordinates

file3 = open(
    os.path.join(dstPath, "Bead_Parameters.txt"), "w"
)  # File for bead parameters

# Write header information for bead parameters in file3
print(
    "\t \t BEAD PARAMETERS: (coordinates), (highslp, highSlpInstances), (prominence), (width), (widthMax) || Next bead",
    file=file3,
)
# Flag to enable saving results
saveResults = True

# %% Parameters for figure drawing
fontScale = 0.35  # Scale for text font
thickness = 2  # Line thickness
font = cv.FONT_HERSHEY_SIMPLEX  # Font type for drawing text
colr = (255, 0, 255)  # Color for drawing annotations
initGuess = [0.001, 0.01, 0.1, 1, 1]  # Initial guesses for some parameters

# %% Load the first image and select the axon
# Read the first image in the series (grayscale)
imFirst = cv.imread(fileIds[k_in], cv.IMREAD_UNCHANGED)

# Convert the grayscale image to a color image for visualization
im_color = cv.cvtColor(zero255(imFirst), cv.COLOR_GRAY2BGR)

# Prompt the user to manually select points on the axon
points = getClick(im_color)

# Set the initial axon diameter
dia = np.float32(15.0)

# Record the start time for performance measurement
startTime = time.time()

# %% Pre-processing of the first image
# Get the dimensions of the first image
h, w = np.shape(imFirst)

# Denoise the first image using wavelet denoising
imFilt = denoise_wavelet(
    imFirst,
    method="BayesShrink",
    wavelet="bior4.4",
    mode="soft",
    wavelet_levels=3,
    rescale_sigma=True,
)

# Pad the filtered image to avoid edge effects
imPad = np.pad(zero255(imFilt), (padVal, padVal), mode="linear_ramp")

# %% Create a mask for the axon
# Initialize a blank mask and skeleton image
mask = np.zeros((h, w), np.uint8)
skltnIni = np.zeros((h, w), np.uint8)

# Draw lines connecting the selected points on the mask and skeleton
for i in range(len(points) - 1):
    cv.line(
        mask, points[i], points[i + 1], color=(255), thickness=30
    )  # Thick mask
    cv.line(
        skltnIni, points[i], points[i + 1], color=(255), thickness=1
    )  # Thin skeleton

# Ensure all non-zero pixels in the mask are set to 255
mask[mask > 0] = 255

# %% Define the region of interest (ROI) for the first image
# ROI is defined in (y1, y2, x1, x2) format based on the selected points
cropBox1 = [
    min(t[1] for t in points),  # Minimum y-coordinate
    max(t[1] for t in points),  # Maximum y-coordinate
    min(t[0] for t in points),  # Minimum x-coordinate
    max(t[0] for t in points),  # Maximum x-coordinate
]

# %% Prepare for template matching
# Initialize axon endpoints and orientation
# axnEnds_pdd -> s_col, s_row, e_col, e_row
axnEnds_pdd, startSlp, endSlp, orient = initAxon(points)

# Define cropping indices for template matching
dummy1 = axnEnds_pdd - templSz  # Start indices
dummy2 = axnEnds_pdd + templSz + 1  # End indices

# Extract templates for the axon ends
templEnd1 = zero255(imPad[dummy1[1]: dummy2[1], dummy1[0]: dummy2[0]])
templEnd2 = zero255(imPad[dummy1[3]: dummy2[3], dummy1[2]: dummy2[2]])

# %% Begin the iteration
try:
    for k in range(
            k_in,
            40):  # len(fileIds)):  # Iterate over the image series
        # %% Pre-process the loaded image
        # Set maximum function evaluations for curve fitting
        maxfev = 1000 if k == 0 else 100
        k_str = "{:03d}".format(
            k
        )  # Format image index as a three-digit string
        im = cv.imread(
            fileIds[k], cv.IMREAD_UNCHANGED
        )  # Load the current image

        # Apply wavelet denoising to the image
        imFilt = denoise_wavelet(
            im,
            method="BayesShrink",
            wavelet=wavelet,
            mode="soft",
            wavelet_levels=wavLev,
            sigma=wavThld * estimate_sigma(im),
            rescale_sigma=True,
        )

        # Pad the filtered image and apply Gaussian blur
        imPad = np.pad(
            cv.GaussianBlur(zero255(imFilt), (0, 0), sigmaX=1.5),
            (padVal, padVal),
            mode="linear_ramp",
        )

        # Convert the original image to color for visualization
        # imColor = cv.cvtColor(zero255(im), cv.COLOR_GRAY2BGR)

        # %% Crop the ends of the axon (for template matching)
        # Define cropping indices for the ends of the axon
        dummy1 = axnEnds_pdd - padVal
        dummy2 = axnEnds_pdd + padVal
        # Crop regions around the axon ends
        snipEnd1 = imPad[dummy1[1]: dummy2[1], dummy1[0]: dummy2[0]]
        snipEnd2 = imPad[dummy1[3]: dummy2[3], dummy1[2]: dummy2[2]]

        # %% Perform multi-angle template matching
        # Match the first end of the axon
        bestMatch = templateMatch_angled(
            snipEnd1, templEnd1
        )  # Returns match in x, y format
        cordsEnd1_pdd = getMatchedCords(
            axnEnds_pdd[0], axnEnds_pdd[1], bestMatch, h, w
        )

        # Match the second end of the axon
        bestMatch = templateMatch_angled(snipEnd2, templEnd2)
        cordsEnd2_pdd = getMatchedCords(
            axnEnds_pdd[2], axnEnds_pdd[3], bestMatch, h, w
        )

        # Remove padding from matched coordinates
        cordsEnd1 = cordsEnd1_pdd - padVal
        cordsEnd2 = cordsEnd2_pdd - padVal

        # %% Check axon tracking
        # Verify if the endpoints found by template matching and curve fitting
        # match
        repeat_tracking = True
        while repeat_tracking:
            colEnd1, rowEnd1 = cordsEnd1  # Extract coordinates (x, y format)
            colEnd2, rowEnd2 = cordsEnd2

            # %% Crop the region of interest (ROI)
            imCrop, imColrCrop, maskCrop, cropShp, cropBox = cropping(
                imFilt, mask, cropBox1, h, w
            )
            hCrop, wCrop = cropShp
            cropY = cropBox[0]  # Start of the cropped region (y-coordinate)
            cropX = cropBox[2]  # Start of the cropped region (x-coordinate)

            # Calculate the start and end points of the axon in the cropped
            # image
            y0 = rowEnd1 - cropY
            yn = rowEnd2 - cropY
            x0 = colEnd1 - cropX
            xn = colEnd2 - cropX

            # Increase the resolution for further processing
            cropShpHr = (
                cropShp[0] * sclFact,
                cropShp[1] * sclFact,
            )  # High-resolution shape
            rHr, cHr = cropShpHr
            maskHr = cv.resize(maskCrop, (cHr, rHr), cv.INTER_NEAREST)
            imHr = cv.resize(imCrop, (cHr, rHr), cv.INTER_CUBIC)

            # Scale coordinates to match the high resolution
            x0Hr = sclFact * x0
            xnHr = sclFact * xn
            y0Hr = sclFact * y0
            ynHr = sclFact * yn

            # %% Enhance contrast using the mask
            pixVals = imHr[
                maskHr.astype(bool)
            ]  # Extract pixel values within the mask
            upper = np.percentile(pixVals, 80)  # Compute the 80th percentile
            imHr[imHr > upper] = (
                upper  # Cap pixel values above the 80th percentile
            )
            imHr = zero255(imHr)  # Normalize bit depth to 8-bit

            # %% Apply Meijering filter to enhance the axon structure
            axon = itkMeijering(image_from_array(imHr.astype(np.int16)))
            axon = zero255(
                axon * maskHr
            )  # Combine the filtered result with the mask

            # %% Binarize the filtered axon image
            _, axonBin = cv.threshold(axon, 15, 255, cv.THRESH_BINARY)
            axonBin = cv.erode(
                axonBin, kd1, iterations=2
            )  # Remove thin connections
            axonBin = binaryFilter(
                axonBin,
                cropShpHr,
                n=int(600 + np.abs(dia - 15.5) * (dia - 15.5) * 100),
            )
            axonBin = cv.dilate(
                axonBin, kd1, iterations=2
            )  # Restore original size
            axonBin = cv.morphologyEx(
                axonBin, cv.MORPH_CLOSE, kd5
            )  # Close gaps

            # %% Remove regions thicker than 2x the diameter
            if k != k_in:  # Skip this step for the first image
                axonBin1 = cv.morphologyEx(axonBin, cv.MORPH_CLOSE, kd20)
                axonBin1 = cv.morphologyEx(
                    axonBin1, cv.MORPH_TOPHAT, disk(int(dia / 2 + 4))
                )
                axonBin1 = cv.morphologyEx(axonBin1, cv.MORPH_OPEN, kd3)
                axonBin1 = binaryFilter(axonBin1, cropShpHr, n=20 * dia)

            # %% Skeletonize the binary axon image
            if k == k_in:  # For the first image, use the initial skeleton
                dummy = skltnIni[
                    cropBox[0]: cropBox[1], cropBox[2]: cropBox[3]
                ]
                skltn = cv.resize(
                    dummy.astype(np.uint8),
                    (cropShpHr[1], cropShpHr[0]),
                    interpolation=cv.INTER_NEAREST,
                )
            else:
                skltn = skeletonize(axonBin1).astype(np.uint8)
                skltn = binaryFilter(skltn, cropShpHr, n=45)

            # %% Curve fitting
            xFit, yFit, slpsHr, initGuess = fitCurve(
                skltn, x0Hr, y0Hr, xnHr, ynHr, orient, initGuess, maxfev
            )

            # %% Check tracking accuracy
            if (
                orient == "horz"
                and (
                    np.abs(yFit[0] - y0Hr) > 90 or np.abs(yFit[-1] - ynHr) > 90
                )
            ) or (
                orient == "vert"
                and (
                    np.abs(xFit[0] - x0Hr) > 90 or np.abs(xFit[-1] - xnHr) > 90
                )
            ):
                # If tracking is inaccurate, ask the user to reselect the axon
                print("Inside correction loop", orient)
                points = getClick(imColor)  # Get new axon points from the user
                mask = np.zeros((h, w), np.uint8)
                for i in range(len(points) - 1):
                    cv.line(
                        mask,
                        points[i],
                        points[i + 1],
                        color=(255),
                        thickness=30,
                    )

                axnEnds_pdd, startSlp, endSlp, orient = initAxon(points)
                cordsEnd1_pdd = axnEnds_pdd[0:2]
                cordsEnd2_pdd = axnEnds_pdd[2:4]
                cordsEnd1 = cordsEnd1_pdd - padVal
                cordsEnd2 = cordsEnd2_pdd - padVal

                cropBox1 = [
                    min(t[1] for t in points),
                    max(t[1] for t in points),
                    min(t[0] for t in points),
                    max(t[0] for t in points),
                ]
                maxfev = 500  # Increase function evaluations for curve fitting
                repeat_tracking = True
            else:
                # Tracking is accurate; exit the loop
                repeat_tracking = False

        # %% Prepare for the next iteration
        # Update the endpoints of the axon for template matching
        axnEnds_pdd = np.concatenate((cordsEnd1_pdd, cordsEnd2_pdd))
        dummy1 = axnEnds_pdd - templSz
        dummy2 = axnEnds_pdd + templSz + 1

        # Extract new templates for the axon ends
        templEnd1 = imPad[dummy1[1]: dummy2[1], dummy1[0]: dummy2[0]]
        templEnd2 = imPad[dummy1[3]: dummy2[3], dummy1[2]: dummy2[2]]

        # %% Generate the axon spine from curve-fitting results
        # Initialize an empty matrix for the spine
        spine = np.zeros(cropShpHr, np.uint8)

        # Populate the spine matrix using the fitted curve
        for y, x in zip(yFit, xFit):
            try:
                spine[y, x] = 1
            except IndexError:
                pass

        # Dilate the spine to create a new mask
        maskFine = cv.dilate(spine, disk(int(np.round(1.5 * dia))))

        # %% Mark the axon for visual display
        # NOTE: This block is only for visualization and does not affect
        # execution
        imColrCrop = cv.circle(
            imColrCrop, (x0, y0), 5, color=(255, 0, 0), thickness=-1
        )
        imColrCrop = cv.circle(
            imColrCrop, (xn, yn), 5, color=(255, 0, 0), thickness=-1
        )

        # %% Update the mask for the next iteration
        maskCrop = cv.resize(
            maskFine,
            (cropShp[1], cropShp[0]),
            interpolation=cv.INTER_NEAREST,
        )
        maskCrop = cv.dilate(maskCrop, kd5)
        mask = mask * 0
        mask[cropBox[0]: cropBox[1], cropBox[2]: cropBox[3]] = maskCrop

        # Define the new cropping area for the next image
        cropBox1 = [
            cropY + int(np.min(yFit) / sclFact),
            cropY + int(np.max(yFit) / sclFact),
            cropX + int(np.min(xFit) / sclFact),
            cropX + int(np.max(xFit) / sclFact),
        ]

        # Update slps for the next iteration
        startSlp = np.mean(slpsHr[:20])
        endSlp = np.mean(slpsHr[-20:])

        # %% Extract edges and skeleton
        # Create edges by dilating the binary axon and subtracting the original
        edges = cv.dilate(axonBin, kd2) - axonBin
        edges = binaryFilter(edges * maskFine, cropShpHr, n=60)

        # %% Calculate axon diameter along the spine ( medial axis)
        orientFlag = np.int8(1) if orient == "horz" else np.int8(-1)
        xs, ys, xe, ye = getPerpendCords(
            xFit, yFit, slpsHr, orientFlag, L=np.uint8(dia + 5)
        )
        dias = calcAxonDia(xs, ys, xe, ye, edges * maskFine, yFit, xFit)
        dias[dias == 0] = np.nan

        # Interpolate and fill missing values in the diameter array
        diasFilled = (
            pd.Series(dias)
            .interpolate(limit=20, limit_direction="forward")
            .ffill()
            .bfill()
            .to_numpy()
        )
        # %%
        axLen = len(diasFilled)

        # %% Denoise the diameter array
        diasFilt = denoiseWavelet_1D(
            diasFilled, "coif4", thldFactor=1.5, level=3
        ).astype(np.float32)
        diasFilt = diasFilt[0:axLen]
        # Calculate the gradient of the diameters
        gradient = np.gradient(diasFilt)

        # %% Calculate the mean/mode of the diameter
        diasInt = np.round(dias[~np.isnan(dias)]).astype(np.uint8)
        diasInt = diasInt[diasInt != 0]
        diaMode = mode(diasInt)[0].astype(np.float32)  # Mode of the diameters
        diasMode = dias[(dias > diaMode - 1.2) & (dias < diaMode + 1.5)]
        dia = np.round(np.mean(diasMode), decimals=2)

        # %%

        prominence = dia / 2.5  # Define prominence threshold
        pks, props = find_peaks(
            diasFilt,
            height=dia + 1,
            threshold=None,
            distance=20,
            prominence=prominence,
            width=(1, 50),
            wlen=6 * dia,
            rel_height=0.5,
            plateau_size=None,
        )

        # Log basic information
        n_pk = len(pks)

        # %% Append data to the result files
        print(">>", "k =", k, "\t dia =", format(dia, ".1f"), file=file1)
        print("\n >>", "k = \t", k, end="", file=file2)
        print("\n >>\t", k, "\t", format(dia, ".1f"), end="", file=file3)

        # %% Rigorous bead analysis
        if n_pk != 0:
            initGuess1 = initGuess[-2:]  # Use quadratic fit for bead analysis
            for i, pos in enumerate(pks):
                # Extract bead properties
                width = int(props["widths"][i])
                pk_x = xFit[pos]
                pk_y = yFit[pos]
                dummyCord = (int(pk_x), int(pk_y))

                # Determine bounds for analyzing bead
                l_ind = int(max(pos - width, 0))
                r_ind = int(min(pos + width, axLen))
                gradsNear = np.abs(gradient[l_ind:r_ind])
                HslpInst = np.sum(gradsNear > 1.75)

                if HslpInst < 4:
                    # Define a wider region for bead analysis
                    width = int(2.4 * dia)
                    beadBin = axonBin[
                        max(pk_y - width, 0): min(pk_y + width, rHr),
                        max(pk_x - width, 0): min(pk_x + width, cHr),
                    ]
                    maskFineB = maskFine[
                        max(pk_y - width, 0): min(pk_y + width, rHr),
                        max(pk_x - width, 0): min(pk_x + width, cHr),
                    ]
                    beadCrop = zero255(
                        imHr[
                            max(pk_y - width, 0): min(pk_y + width, rHr),
                            max(pk_x - width, 0): min(pk_x + width, cHr),
                        ]
                    )

                    # Annotate the bead position in the visual output
                    cv.circle(
                        imColrCrop,
                        (int(pk_x / sclFact), int(pk_y / sclFact)),
                        10,
                        (0, 255, 255),
                        1,
                    )
                    cv.putText(
                        imColrCrop,
                        str(dummyCord),
                        (int(pk_x / sclFact - 20), int(pk_y / sclFact - 15)),
                        font,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv.LINE_8,
                    )
                    # Log basic bead properties
                    print(
                        "\t i= ",
                        i,
                        dummyCord,
                        " Initial slope =",
                        format(HslpInst, ".2f"),
                        "\t",
                        file=file1,
                    )

                    # Rigorous bead analysis
                    try:
                        propsB, cnfB = rigBeadAnal(
                            beadCrop,
                            beadBin * maskFineB,
                            dia,
                            orient,
                            initGuess1,
                            file1,
                            k,
                            i,
                            dstPath,
                        )

                        if propsB:
                            print(
                                "\t",
                                dummyCord,
                                f"({propsB[0][0]:.2f}, {propsB[0][1]})",
                                f"({propsB[1]:.2f})",
                                f"({propsB[2]:.2f})",
                                f"({propsB[3]:.2f})",
                                "||",
                                end="",
                                file=file3,
                            )
                            if cnfB:
                                print(
                                    "\t",
                                    int(pk_x),
                                    int(pk_y),
                                    end="",
                                    file=file2,
                                )
                    except Exception:
                        print(
                            f"Error occurred during bead analysis at k={k} and i={i}"
                        )

        # Log the end of the current iteration
        print(f"__________End of k = {k}___________")

        # %% Bokeh Plotting
        if opDetails:
            # Convert the cropped image to a format compatible with Bokeh
            imColrCrop1 = image2BokehFormat(imColrCrop)

            # Initialize a blank array for overlaying edges and masks
            dummy = np.zeros(cropShpHr)
            edges1 = np.dstack((spine, edges, maskFine / 4))

            # Create a mask for the background where all three channels are
            # zero
            background_mask = (
                (edges1[:, :, 0] == 0)
                & (edges1[:, :, 1] == 0)
                & (edges1[:, :, 2] == 0)
            )

            # Set background pixels to a fixed value (e.g., 200 for
            # visualization)
            edges1[background_mask] = 200
            edges1 = image2BokehFormat(edges1)

            # Create the title as an HTML string for display
            title_text = (
                "<div style='text-align: center;'><h1>____________________________Frame No. = " +
                str(k) +
                "</h1></div>")
            title_div = Div(text=title_text, width_policy="fit")

            # Add a divider at the bottom to display the source path
            bottom_line = Div(
                text=f"<div style='border-top: 2px solid black; margin-top: 5px;'>{srcPath}</div>",
                width=1300,
            )

            # Prepare data for plotting the diameter graph
            x = np.arange(len(dias))  # x-axis: distance along the axon
            data = {
                "x": x,
                "y": diasFilt,  # Filtered diameter values
                "pixel_x": xFit,  # Pixel x-coordinates
                "pixel_y": yFit,  # Pixel y-coordinates
            }
            source = ColumnDataSource(data)

            # Create a figure for the diameter plot
            p = figure(width=1600, height=600)
            p.title = Title(
                text="Mode Dia = " + str(dia),
                align="center",
                text_font_size="14pt",
            )
            p.line("x", "y", source=source)  # Line plot of diameter values
            p.scatter(
                "x",
                "y",
                source=source,
                fill_color="red",
                size=4,
                marker="circle",
            )  # Overlay scatter points

            # Customize axis labels and font sizes
            p.xaxis.axis_label = (
                "Along the axon \n (from one end to the other)"
            )
            p.xaxis.axis_label_text_font_size = "15pt"
            p.yaxis.axis_label = "Dia of the axon"
            p.yaxis.axis_label_text_font_size = "15pt"
            p.xaxis.major_label_text_font_size = "12pt"
            p.yaxis.major_label_text_font_size = "12pt"

            # Add a hover tool for the scatter plot
            scatter_hover = HoverTool(
                tooltips=[
                    ("X", "@pixel_x"),
                    ("Y", "@pixel_y"),
                    ("(x,y)", "($x, $y)"),
                ]
            )
            p.add_tools(scatter_hover)

            # Create figures for displaying the images
            fig1 = figure(
                width=800,
                height=700,
                x_range=(0, cropShp[1]),
                y_range=(cropShp[0], 0),
                match_aspect=True,
            )
            fig1.image_rgba(
                image=[imColrCrop1],
                x=0,
                y=0,
                dw=cropShp[1],
                dh=cropShp[0],
            )
            fig1.title = Title(
                text="Original Image", align="center", text_font_size="14pt"
            )
            fig1.sizing_mode = "fixed"

            fig2 = figure(
                width=800,
                height=700,
                x_range=(0, cropShpHr[1]),
                y_range=(cropShpHr[0], 0),
                match_aspect=True,
            )
            fig2.image_rgba(
                image=[edges1],
                x=0,
                y=0,
                dw=cropShpHr[1],
                dh=cropShpHr[0],
            )
            fig2.title = Title(
                text="Edge Detection", align="center", text_font_size="14pt"
            )
            fig2.sizing_mode = "fixed"

            # Add hover tools to display coordinates on the images
            image_hover = HoverTool(tooltips=[("x", "$x"), ("y", "$y")])
            fig1.add_tools(image_hover)
            fig2.add_tools(image_hover)

            # Combine the two image figures into a single row
            top_row = row(fig1, fig2)

            # Add the diameter plot below the image figures
            bottom_row = p

            # Spacer to adjust layout padding
            spacer = Spacer(width=800, height=20)

            # Combine all components into a vertical layout
            layout = column(
                title_div, spacer, top_row, bottom_row, bottom_line
            )

            # Save the layout as an HTML file
            full_path = os.path.join(dstPath, k_str + "_analysis.html")
            output_file(full_path)
            save(layout)

# %% Handle interruptions and save results if the code stops
except Exception as e:
    print(f"An error occurred: {e}")
    # Prompt the user to decide whether to save results processed so far
    user_decision = (
        input("Do you want to save results processed so far? (y/n): ")
        .strip()
        .lower()
    )
    if user_decision == "y":
        print("Proceeding to save results.")
    else:
        saveResults = False
        print("Skipping result saving.")
        raise  # Re-raise the exception to terminate execution if required

# Close files if results are saved
if saveResults:
    file1.close()
    file2.close()
    file3.close()

# Log the total runtime
endTime = time.time()
print(f"Time taken by the code to run: {endTime-startTime} seconds")
