#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This checks  start of beding once the data on beds at each frame is avilable


Created on Wed Sep  4 15:37:39 2024

@author: pretheesh
"""

import os
import pandas as pd
from openpyxl import Workbook
import numpy as np
from beading_functions_21 import selectFolder
#%%
n_frames =90
# Define the base directory containing the subfolders
folderPath = selectFolder('Select the folder containing the analysis results')
# Define the output Excel file path
output_xls = '/mnt/W_D_Debian/py_Dst/3000-2'
#%% Collect beading related files from all subfolders
allFiles = []  # List to collect matching file paths
for folderName in sorted(os.listdir(folderPath)):
    path = os.path.join(folderPath, folderName)
    if os.path.isdir(path):  # Ensure it's a directory
        for fileName in os.listdir(path):
            if fileName.endswith('Bead_Cords.txt'):
                beadCords = os.path.join(path, fileName)
                allFiles.append(beadCords)  # Add to the list
#%%
beading_onset = []
# Initialize an Excel writer object
with pd.ExcelWriter(output_xls, engine='openpyxl') as writer:
    # Create a dummy sheet to avoid "no sheet visible" issue
    dummy_sheet_name = "DummySheet"
    writer.book.create_sheet(dummy_sheet_name)
    # Track if any valid sheets are created
    valid_sheet_created = False
    
    for i, beadCords in enumerate(allFiles):              
        data = []
        with open(beadCords, 'r') as file:
            for line in file:
                 # Split the line by tabs
                parts = [part.strip() for part in line.split('\t') if part.strip()]
                data.append(parts[1:])
               
            for j in range(len(data) - 3):
                # Take four consecutive rows
                rowSequence = data[j:j + 4]                                              
                # Count non-empty columns in each row
                counts = [len(row) for row in rowSequence]
                
                if sum(counts) >= 7:
                    # Find the first row with length > 1
                    for row in rowSequence:
                        if len(row) > 1:
                            # Append the first element of the first row with length > 1
                            beading_onset.append(int(row[0]))
                            beading_onset = [item for item in beading_onset if item != 0]
                            break
                    break
                elif j == n_frames - 3:#reached till end without any onset
                    beading_onset.append('NB')                   
            if data:
                df = pd.DataFrame(data)

                # Write the DataFrame to a new sheet in the Excel file
                sheet_name = f'Sheet{i + 1:02d}'  # Name sheets as Sheet01, Sheet02, etc.
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

                valid_sheet_created = True
                  

    # Remove the dummy sheet if any valid sheets were created
    if valid_sheet_created:
        del writer.book[dummy_sheet_name]

print(f"Data successfully written to {output_xls}")
#%%
x= [0,10, 20, 30, 40, 50, 60, 70, 80, 90]
# Calculate the histogram and cumulative frequency
nb_count = beading_onset.count('NB')
beading_onset1=[item for item in  beading_onset if item != 'NB']
hist,_= np.histogram(beading_onset1, bins=9, range=(0,90))
cumlFreq = np.cumsum(hist)*100/(len(beading_onset))
cumlFreq=np.insert(cumlFreq, 0, 0)
np.save('500cumul',cumlFreq)
#%%
#
human_3000_nM = [0, 62.82, 93.05, 97.43, 100, 100, 100, 100, 100, 100]
human_500_nM = [0, 27.27, 58.18, 85.45, 94.54, 96.36, 96.36, 96.36, 96.36, 96.36]
human_100_nM = [0, 23.37, 46.75, 66.23, 80.52, 85.71, 88.31, 89.61, 90.91, 92.2]

x= [0,10, 20, 30, 40, 50, 60, 70, 80, 90]


cumlFreq3000 = np.load('3000cumul.npy')
cumlFreq500 = np.load('500cumul.npy')
cumlFreq100 = np.load('100cumul.npy')
#


#%%
import plotly.graph_objects as go
from plotly.express import imshow
import plotly.express as px

import plotly.io as io
io.renderers.default='browser'
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=x,
                          y=cumlFreq3000,
                          mode='lines+markers',
                          name='Machine_3000nM',
                          marker=dict(size=10)))  # Add a legend entry name
fig.add_trace(go.Scatter(x=x,
                          y=human_3000_nM,
                          mode='lines+markers',
                          name='human_3000_nM',
                          marker=dict(size=10)))  # Add a legend entry name
fig.add_trace(go.Scatter(x=x,
                          y=human_500_nM,
                          mode='lines+markers',
                          name='human_500_nM',
                          marker=dict(size=10)))
fig.add_trace(go.Scatter(x=x,
                          y=cumlFreq500,
                          mode='lines+markers',
                          name='Machine_500nM',
                          marker=dict(size=10)))  
fig.add_trace(go.Scatter(x=x,
                          y=human_100_nM,
                          mode='lines+markers',
                          name='human_100_nM',
                          marker=dict(size=10)))
fig.add_trace(go.Scatter(x=x,
                          y=cumlFreq100,
                          mode='lines+markers',
                          name='Machine_100nM',marker=dict(size=10))) 
#Update layout for axis titles and legend title
fig.update_layout(
    title="Vinristine: Human vs Machine",  # Title of the entire plot
    xaxis_title="Time in minutes",  # Title of the x-axis
    yaxis_title="Percentage of axons ",  # Title of the y-axis
   legend=dict(
        x=0.5,  # Horizontal position (centered)
        y=0.5,  # Vertical position (middle)
        xanchor='center',  # Anchor point for x
        yanchor='middle',  # Anchor point for y
        bgcolor="rgba(255, 255, 255, 0.8)",  # Background color with transparency
        bordercolor="black",  # Border color
        borderwidth=1  # Border width
    ),
    font=dict(
        family="Arial, sans-serif",
        size=20,
        color="black"
    )
)

# Show the figure
fig.show()

#%%
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'  # Ensure fonts are embedded as text
plt.figure(figsize=(5 ,5),dpi=300)

# Plot each dataset with markers and lines
plt.plot(x, human_3000_nM, label='human - 3000 nM', marker='s', markersize=8,linestyle='-.', linewidth=2)
plt.plot(x, cumlFreq3000, label='machine - 3000 nM', marker='o', markersize=8,linestyle='-.', linewidth=2)

plt.plot(x, human_500_nM, label='human - 500 nM', marker='s', markersize=8,linestyle='-', linewidth=2)
plt.plot(x, cumlFreq500, label='machine - 500 nM', marker='o', markersize=8,linestyle='-', linewidth=2)
plt.plot(x, human_100_nM, label='human - 100 nM', marker='s', markersize=8,linestyle='--', linewidth=2)
plt.plot(x, cumlFreq100, label='machine - 100 nM', marker='o', markersize=8,linestyle='--', linewidth=2)
 
# Add title and labels
#plt.title("Vinristine: Human vs Machine", fontsize=16)
plt.xlabel("time in minutes", fontsize=11)
plt.ylabel("percentage of axons beaded", fontsize=11)
plt.tick_params(axis='x', labelsize=10)  # Control x-axis tick size
plt.tick_params(axis='y', labelsize=10)  # Control y-axis tick size
# Customize legend and position it inside the graph
plt.legend(
    loc='center',  # Predefined location (center of the plot)
    bbox_to_anchor=(0.72, 0.35),  # Precise location in the plot area
    frameon=False,  # Enable a legend frame
    handlelength=5,         # Length of the legend line
    handleheight = 1.1,         # Height of the legend marker
    markerscale = 1,          # Scale factor for markers in the legend
    borderpad = 1,              # Padding inside the legend box
    labelspacing = 1.1,         # Spacing between legend entries
    facecolor = 'white',  # Legend background color
    edgecolor = 'black' , # Legend border color
    fontsize = 10
)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

