# **Analysis of Spatial and Temporal Diameter Fluctuations in Degenerating Axons**

Axonal beading, a shape change in axons resulting in a modulation of their diameter, is often associated with **axonal degeneration**. This repository provides a **Python-based tool** for studying axonal beading that occurs due to the **neurotoxic effects of chemotherapy drugs**, such as **Vincristine**.

This tool iterates through a **time series of axonal images** treated with Vincristine, identifying the **onset of beading**, which serves as a **marker for neurotoxicity**. Additionally, it provides **precise measurements** of various axonal metrics over time, including:

- **Spatial and temporal fluctuations** of the axon during degeneration  
- **Evolution of axonal diameter** at the bead’s terminations  
- **Changes in bead diameter and prominence** over time  
- **Tracking the movement and merging of beads** along the axon  

Detailed description of the algorithm and implementation can be found in the manuscript: https://doi.org/10.1101/2025.02.05.636573

---

## **Authors & Acknowledgments**
This project was developed by:

- **Pretheesh Kumar V C**  
- **Pramod Pullarkat**  

at the **Cell Biophysics Lab, Raman Research Institute (RRI), Bangalore, India**. We acknowledge the generous support from **The Wellcome Trust-DBT India Alliance** (Grant **IA/TSG/20/1/600137**) and **RRI**, which enabled the development of this research tool.

---

## **File Structure**
The repository contains the following key files and folders:

1. **`configuration_beading_onset.yaml`**  
   - A configuration file where parameters can be modified using any text editor.

2. **`tracker_beading_onset.py`**  
   - The **main script** that processes the axonal image series.

3. **`functions_beading_onset.py`**  
   - A script containing all the necessary **functions** used for tracking beading onset.

4. **`100_nM_Vincristine/`**  
   - A folder containing a set of **sample images** that can be used to run and test the code.

5. **`pyDst/`**  
   - The **default output folder** where processed results will be saved.

6. **`requirements.txt`**  
   - A file listing all required packages. Use this to set up the environment:
     ```bash
     pip install -r requirements.txt
     ```

> **Note**:  
> - The code was developed using **Python 3.10.12** on **Linux** and **Python 3.10.11** on **Windows**.

---

## **How to Run the Code**
1. **Set up the environment**:
   ```bash
   pip install -r requirements.txt


1. **Run the main script**:
   ```bash
   python tracker_beading_onset.py
   ```
2. **Selecting the Image Folder**

   Select the image folder when prompted.

> **Note:**\
> By default, the script uses a hardcoded path:   `srcPath = "100_nM_Vincristine"`  
> To enable **interactive folder selection**, replace it - around **line 15** in `tracker_beading_onset.py` -  with :  
> ```python
> srcPath = selectFolder("Select the folder with the image series")
4. **Manually select an axon** in the displayed image window:
   - Click **from one end to the other continuously** (without backtracking).
   - left click to add a point and right click to remove
5. **Press any key** to continue.
6. The code will iterate through all images, and the results will be **stored** in the path specified by the `"dstPath"` field in the **`configuration_beading_onset.yaml`** file.
> **Note:**  
> If you are using a **Windows N edition**, some required `.dll` files for **OpenCV** may be missing. To fix this:
> 1. Go to **Settings → Apps → Optional Features → Add a feature**  
> 2. Search for **Media Feature Pack** and install it  
> 3. **Restart** your system



---

## **Output**
- The script **automatically processes** the entire time series and generates **quantitative measurements** of axonal beading.  
- The output files contain **detailed tracking results** for further analysis.  
- If the field `"opDetails"` in the **`configuration_beading_onset.yaml`** is set to `"true"`, the output folder will contain a **detailed result**. Otherwise, three key output files will be generated:
  
  1. **`Bead_Cords.txt`** - Contains the coordinates of detected beads in each frame.
  2. **`Bead_Parameters.txt`** - Stores the features of all detected beads.
  3. **`Rig_Bead.txt`** - Lists all suspected beads along with the reasons they were not classified as valid beads.

---

