# Semi-automated analysis of beading in degenerating axons #
Axonal beading, which is a shape change of axons
resulting in a modulation of axonal diameter, this often occurs as a result of axonal degenrtation.
This is a code for studying the beading that occurs as a results of the neurotoxic effects ofchemo therapetic drugs (for e.g. Vincristine).
It iterates through a time series of images of axon treated with Vincristine and identifies the onset of beading which a marker for the neurotoxicity of the drug.
THe code can also provide  precise measurements of various axonal metrics over time as the axon deteriorates. These include
the spatial and temporal fluctuations of axon over time, the evolution of axon diameter at the bead’s terminations,
changes in the bead’s diameter and prominence, and the movement and merging of beads along the axon.

The code contains three files
1. configuration_beading_onset.yaml - can be opened in any standard text editor and the parameters can be changed.
2. tracker_beading_onset.py - the main script.
3. functions_beading_onset.py - the file with all the functions written.
How to run
The user can 
