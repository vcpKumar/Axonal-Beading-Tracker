# **Analysis of Spatial and Temporal Diameter Fluctuations in Degenerating Axons**

Axonal beading, a shape change in axons resulting in a modulation of their diameter, is often associated with **axonal degeneration**. This repository provides a **Python-based tool** for studying axonal beading that occurs due to the **neurotoxic effects of chemotherapy drugs**, such as **Vincristine**.

This tool iterates through a **time series of axonal images** treated with Vincristine, identifying the **onset of beading**, which serves as a **marker for neurotoxicity**. Additionally, it provides **precise measurements** of various axonal metrics over time, including:

- **Spatial and temporal fluctuations** of the axon during degeneration.  
- **Evolution of axonal diameter** at the bead’s terminations.  
- **Changes in bead diameter and prominence** over time.  
- **Tracking the movement and merging of beads** along the axon.  

---

## **Authors & Acknowledgments**
This project was developed by:

- **Pretheesh Kumar V C**
- **Pramod Pullarkat**  

at the **Cell Biophysics Lab, Raman Research Institute (RRI), Bangalore, India**. We acknowledge the generous support from **The Wellcome Trust-DBT India Alliance** (Grant **IA/TSG/20/1/600137**) and **RRI**, which enabled the development of this research tool.

---

## **File Structure**
The repository contains the following key files:

1. **`configuration_beading_onset.yaml`**  
   - A configuration file where parameters can be modified using any text editor.

2. **`tracker_beading_onset.py`**  
   - The **main script** that processes the axonal image series.

3. **`functions_beading_onset.py`**  
   - A script containing all the necessary **functions** used for tracking beading onset.

4. **`100_nM_Vincristine/`**  
   - A folder containing a set of **sample images** that can be used to run and test the code.
Full details of the algorithm can be found in the manuscript, https://doi.org/10.1101/2025.02.05.636573
---

## **How to Run the Code**
1. **Run the main script**:
   ```bash
   python tracker_beading_onset.py
   ```
2. **Select the folder** containing the image series when prompted.
3. **Manually select an axon** in the displayed image window:
   - Click **from one end to the other continuously** (without backtracking).
4. **Press any key** to continue.
5. The code will iterate through all images, and the results will be **stored** in the path specified by the `"dstPath"` field in the **`configuration_beading_onset.yaml`** file.

---

## **Output**
- The script **automatically processes** the entire time series and generates **quantitative measurements** of axonal beading.  
- The output files contain **detailed tracking results** for further analysis.  
- If the field `"opDetails"` in the **`configuration_beading_onset.yaml`** is set to `"true"`, the output folder will contain a **detailed result**. Otherwise, three key output files will be generated:
  
  1. **`Bead_Cords.txt`** - Contains the coordinates of detected beads in each frame.
  2. **`Bead_Parameters.txt`** - Stores the features of all detected beads.
  3. **`Rig_Bead.txt`** - Lists all suspected beads along with the reasons they were not classified as valid beads.

---

