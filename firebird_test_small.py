import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *

print(get_diff_interactions(1.3))

make_heatmap_confidence_interval_plot(3, 400, 10)
print("first heatmap done")
