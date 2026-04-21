import importlib
lunar_neutrino_lib_copy = importlib.import_module("lunar_neutrino_lib_copy")
importlib.reload(lunar_neutrino_lib_copy)
from lunar_neutrino_lib_copy import *
"""
ratios = [-50, 0, 50] # mus
radii = [1, 10, 19] # sigma"""
ratios = [0, 50] # mus
radii = [10, 19] # sigma

for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 100)
        print("made a heatmap 10 neutrinos")


print_timing_summary()