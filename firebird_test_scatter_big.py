import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *
ratios = [3, 5, 14,]
radii = [200, 400, 600]
for ratio in ratios:
    for radius in radii:
        make_scatter_plot(ratio, radius, 100, 100)
        print("Made scatter plot 100 neutrinos")

for ratio in ratios:
    for radius in radii:
        make_scatter_plot(ratio, radius, 1000, 100)
        print("Made scatter plot 1000 neutrinos")


