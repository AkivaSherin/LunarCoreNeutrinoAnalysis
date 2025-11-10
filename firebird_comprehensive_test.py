import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *

ratios = [2, 7, 14]
radii = [250, 400, 550]
for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 10)
        print("made a heatmap 10 neutrinos")

for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 100)
        print("made a heatmap 100 neutrinos")

for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 10000)
        print("made a heatmap 10000 neutrinos")

