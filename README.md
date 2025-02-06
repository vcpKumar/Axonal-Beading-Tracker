# **Semi-Automated Analysis of Beading in Degenerating Axons**

Axonal beading, a shape change in axons resulting in a modulation of their diameter, is often associated with **axonal degeneration**. This repository provides a **Python-based tool** for studying axonal beading that occurs due to the **neurotoxic effects of chemotherapy drugs**, such as **Vincristine**.

This tool iterates through a **time series of axonal images** treated with Vincristine, identifying the **onset of beading**, which serves as a **marker for neurotoxicity**. Additionally, it provides **precise measurements** of various axonal metrics over time, including:

- **Spatial and temporal fluctuations** of the axon during degeneration.  
- **Evolution of axonal diameter** at the bead’s terminations.  
- **Changes in bead diameter and prominence** over time.  
- **Tracking the movement and merging of beads** along the axon.  

---

## **File Structure**
The repository contains the following key files:

1. **`configuration_beading_onset.yaml`**  
   - A configuration file where parameters can be modified using any text editor.

2. **`tracker_beading_onset.py`**  
   - The **main script** that processes the axonal image series.

3. **`functions_beading_onset.py`**  
   - A script containing all the necessary **functions** used for tracking beading onset.

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

---

