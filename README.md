# Semi-Automated Analysis of Beading in Degenerating Axons

This code is associated with the paper titled **"Semi-automated analysis of beading in degenerating axons"**, produced as part of the research conducted at the **Cell Biophysics Lab** of the **Raman Research Institute**, Bangalore, India, under the guidance of **Prof. Pramod Pullarkat**. (The details of the paper will be added shortly)
## Overview

The code is designed to track the onset of morphological changes, specifically the beading of axons, in a time series of images. This is achieved by measuring the diameter of the axon at each pixel along its medial axis. Peaks in the diameter indicate potential beading.

For each peak, the code performs a rigorous check to confirm it as a bead. For further details on the methodology and validation, please refer to the associated paper.
## Input and Configuration
### Input

The input to the code is a folder with a time series of axon images.
### Configuration

The analysis parameters can be customized in the configuration file:
configuration_beading_onset.yaml

## Usage Instructions
### Running the Script
**Execute the script:**     
Executing the scrit, *_tracker_beading_onset.py_*, initializes the code. Make sure all three files are in same folder

**Folder Selection:**   

Upon running the script, a prompt will appear for selecting the folder containing axon images.

**Axon Selection:** 

Select an axon by clicking two or more points along its path (if the axon is curved).
Note: Ensure the points are clicked in a continuous sequence from one end of the axon to the other.
 
**Analysis and Results:** 

The code will process the selected axon and analyze the images.
Results will be saved in a separate directory specified in the destination_path field of the configuration file.

## Key Features

**Time-Series Analysis:** 
Tracks the morphological changes in axons over time.

**Diameter Measurement:** 
Computes the diameter of the axon along its medial axis.

**Beading Detection:** 
Identifies and rigorously verifies peaks in the diameter that correspond to beading.

**Configurable Parameters:** 
Provides flexibility to adjust parameters through the YAML configuration file.

## Notes

While selecting axon points, ensure continuity to maintain accuracy.

Refer to the associated research paper for detailed insights into the analysis process.
