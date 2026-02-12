import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *
ratios = [10, 14]
radii = [400, 550]

for ratio in ratios:
    for radius in radii:
        make_scatter_plot(ratio, radius, 1000, 1000)
        print("Made scatter plot 1000 neutrinos")

for ratio in ratios:
    for radius in radii:
        make_scatter_plot(ratio, radius, 10000, 1000)
        print("Made scatter plot 10000 neutrinos")
