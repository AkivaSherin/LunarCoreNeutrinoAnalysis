import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *

print(get_diff_interactions(1.3))

ratios = [5, 10]
radii = [250, 500]
for ratio in ratios:
    for radius in radii:
        make_heatmap_confidence_interval_plot(ratio, radius, 100)
        print("made a heatmap 100 neutrinos")