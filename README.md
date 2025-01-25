# tool for precise estimation diameter of ridge like structures such as neurons, blood vessels, road networks and geographical features

This code was written primarily for the semi-automated analysis of morphological changes occuring in axons of neuronal cells. The method is discussed in detail in the manuscript titled **"Semi-automated analysis of beading in degenerating axons"** (the details of the paper will be added shortly). This code was developed at the **Cell Biophysics Lab** of the **Raman Research Institute**, Bangalore by  **Dr. Pretheesh Kumar V C** and **Prof. Pramod Pullarkat**. This work was funded by the 'Raman Research Institute' and 'the Wellcome Trust DBT India Alliance (grant IA/TSG/20/1/600137)'. 
## Overview

Primarily, the code is designed to track the onset of morphological changes, specifically the beading of axons, in a time series of microscope images. Beading refers to a appearance of modulations in diameter (a series of swellings) along axons. These modulations, which usually is a sign of axonal degeneration, can be mimmicked in the lab using neuronal cultures. 

*This code is designed to overcome the subjectivity and reduce the human effort when analyzing large number of images of beaded axons.* 

The code determines the number and position of beads (swellings) along axons selected by the user.This is achieved by measuring the diameter of an axon along its medial axis. Peaks in the diameter indicate potential beading.  A set of criteria are defined within the code to accurately detect beads and to reject artefacts. 

*_This code can also be applied to analyse diameters of ridge like structures (blood vessels, road networks, geographical features, etc)._* 

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
