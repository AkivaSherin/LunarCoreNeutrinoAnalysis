import importlib
lunar_neutrino_lib = importlib.import_module("lunar_neutrino_lib")
importlib.reload(lunar_neutrino_lib)
from lunar_neutrino_lib import *
import numpy as np
import matplotlib.pyplot as plt

plt.figure()

ratios = [1, 5, 15]
fixed_radius = 600
log_energies = np.linspace(-1,1, 100) # log GeV

for ratio in ratios:
    pdf = muon_pdf_spline(ratio, fixed_radius)
    plt.plot(log_energies, pdf(log_energies), label=str(ratio))

plt.xlabel("Energy (log GeV)")
plt.ylabel("Probability")
plt.title("Muon detection PDF for core radius = " + str(fixed_radius) + " km")
plt.legend(title="Density Ratios")
plt.savefig("muon_detection_pdfs_by_ratios_radius" + str(fixed_radius) + ".png")

plt.figure()

ratios = [1, 5, 15]
fixed_radius = 400

for ratio in ratios:
    pdf = muon_pdf_spline(ratio, fixed_radius)
    plt.plot(log_energies, pdf(log_energies), label=str(ratio))

plt.xlabel("Energy (log GeV)")
plt.ylabel("Probability")
plt.title("Muon detection PDF for core radius = " + str(fixed_radius) + " km")
plt.legend(title="Density Ratios")
plt.savefig("muon_detection_pdfs_by_ratios_radius" + str(fixed_radius) + ".png")


plt.figure()

fixed_ratio = 5
radii = [200, 400, 600]

for radius in radii:
    pdf = muon_pdf_spline(fixed_ratio, radius)
    plt.plot(log_energies, pdf(log_energies), label=str(radius))

plt.xlabel("Energy (log GeV)")
plt.ylabel("Probability")
plt.title("Muon detection PDF for core ratio = " + str(fixed_ratio))
plt.legend(title="Radii (km)")
plt.savefig("muon_detection_pdfs_by_radii_ratio" + str(fixed_ratio) + ".png")

plt.figure()

fixed_ratio = 15
radii = [200, 400, 600]

for radius in radii:
    pdf = muon_pdf_spline(fixed_ratio, radius)
    plt.plot(log_energies, pdf(log_energies), label=str(radius))

plt.xlabel("Energy (log GeV)")
plt.ylabel("Probability")
plt.title("Muon detection PDF for core ratio = " + str(fixed_ratio))
plt.legend(title="Radii (km)")
plt.savefig("muon_detection_pdfs_by_radii_ratio" + str(fixed_ratio) + ".png")