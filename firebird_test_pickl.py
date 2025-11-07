import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *

pdf = muon_pdf_spline(13, 421)

print("found/made pdf")
