import importlib
lunar_neutrino_lib_copy = importlib.import_module("lunar_neutrino_lib_copy")
importlib.reload(lunar_neutrino_lib_copy)
from lunar_neutrino_lib_copy import *
ratios = [-5, 0, 5]
radii = [0, 10, 100]

for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 10)
        print("made a heatmap 10 neutrinos")
"""
for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 100)
        print("made a heatmap 100 neutrinos")
"""
"""for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 1000)
        print("made a heatmap 10000 neutrinos")"""
