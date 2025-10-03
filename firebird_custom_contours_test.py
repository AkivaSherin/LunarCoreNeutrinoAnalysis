import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *

ratios = [5, 10, 30]
radii = [100, 400]
for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 10, percentiles=(66, 90, 99))
        print("made a heatmap 100 neutrinos")

for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 10, percentiles=(10, 60, 99.5))
        print("made a heatmap 100 neutrinos")