import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *
import numpy as np
import matplotlib.pyplot as plt

plt.figure()

ratios = [1, 5, 15]
radius = 400
log_energies = np.linspace(0,0.99, 100) # log GeV

for ratio in ratios:
    pdf = muon_pdf_spline(ratio, radius)
    plt.plot(log_energies, pdf(log_energies), label=fr"ρ_c / ρ_m = {ratio:g}")

plt.xlabel("Energy ")
plt.ylabel("Probability")
plt.title("Muon detection PDF for core radius = " + str(radius))
plt.legend(title="Density Ratios")
plt.savefig("muon_detection_pdfs_by_ratios.png")