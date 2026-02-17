import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
import math
import pandas as pd
from scipy.interpolate import interp1d
import pickle
from scipy.interpolate import griddata
from scipy.integrate import odeint, quad

"""

    Neutrino flavor oscillation simulation

"""


# models neutrino travel to earth using approximation form gaisser textbook
def decoherence_travel_earth(e_prob, m_prob, t_prob):
    p = ([.55, .25, .2],
         [0.25, 0.37, 0.38],
         [0.2, 0.38, 0.42])
    initial_ratios = ([e_prob, m_prob, t_prob])
    product = np.matmul(p, initial_ratios)
    return product


# this returns a density function that gives lunar density as a function of x, y, z for a given core-mantle density ratio
# x y and z are in km, and the center of the moon is 0,0,0
def make_density_function_from_ratio(core_mantle_density_ratio, core_radius):  # core radius in km
    r_moon = 1737 - 10  # km
    r_moon_cm = r_moon * 10 ** 5
    crust_to_core_distance = 1700 * 10 ** 5  # cm
    r_core_cm = core_radius * 10 ** 5  # cm

    m_moon = 7.342 * 10 ** 25  # grams
    m_crust = (4 / 3) * math.pi * (r_moon_cm ** 3 - crust_to_core_distance ** 3)
    m_moon_without_crust = m_moon - m_crust
    V_core = (4 / 3) * math.pi * (r_core_cm ** 3)
    V_mantle = (4 / 3) * math.pi * (crust_to_core_distance ** 3 - r_core_cm)

    mantle_density = m_moon_without_crust / (V_core * core_mantle_density_ratio + V_mantle)
    core_density = mantle_density * core_mantle_density_ratio

    def density_function(x, y, z, d_core=core_density, d_mantle=mantle_density):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r < core_radius:  # core
            # print("core density" + str(d_core))
            return d_core
        if r < 1700:  # mantle
            # print("mantle density" + str(d_mantle))
            return d_mantle
        else:
            return 2.55  # crust density is known pretty accurately

    return density_function


# convers m to e nue ratio into list of probability amplitudes that we can use in our oscillation code
# for now we assume 2 muon : 1 electron ratio
def turn_flavor_ratios_into_probability_amplitudes(m_to_e_ratio):
    # set initial amplitudes
    m_amp = m_to_e_ratio
    e_amp = 1
    # normalize
    initial_magnitude = math.sqrt(m_amp ** 2 + e_amp ** 2)

    m_amp = m_amp / initial_magnitude
    e_amp = e_amp / initial_magnitude

    return [e_amp, 0, m_amp, 0, 0, 0]


# this is modified code given to me by Prof. Hyde
# simulates neutrino oscillation, but specialized for moon with variable density
# y, z are fixed values, while neutrino travels parallel to x-axis
# density function takes x, y, z (location relative to center of moon) and returns moon's density in g/cm^3

# gives oscillation derivative, but specialized for moon.
# x, y, z is location relative to center of moon
# density function takes this location and returns a density

# Define mass-squared difference (in eV^2) and mixing angle (rad)
Th12 = 34.5 * np.pi / 180
Th13 = 8.45 * np.pi / 180
Th23 = 47.7 * np.pi / 180  # assuming normal mass ordering
Dmsq21 = 7.55 * 10 ** (-5)
Dmsq31 = 2.50 * 10 ** (-3)  # assuming normal mass ordering


def moon_deriv_three_flavor(amp, x, E_GeV, del13, y, z, density_function):
    # Set mass values
    mass1 = 0.0
    mass2 = Dmsq21
    mass3 = Dmsq31

    ## Elements of flavor-basis Hamiltonian

    # Mixing matrix (unitary change of basis between mass and flavor)
    # Sines and Cosines of Mixing Angles
    s12 = np.sin(Th12)
    c12 = np.cos(Th12)
    s23 = np.sin(Th23)
    c23 = np.cos(Th23)
    s13 = np.sin(Th13)
    c13 = np.cos(Th13)

    ## Elements of flavor-basis Hamiltonian

    # Mixing matrix
    U12 = np.array([(c12, s12, 0), (-s12, c12, 0), (0, 0, 1)])
    U23 = np.array([(1, 0, 0), (0, c23, s23), (0, -s23, c23)])
    U13 = np.array([(c13, 0, s13 * np.exp(0. - 1.j * del13)), (0, 1, 0), (-s13 * np.exp(0. + 1.j * del13), 0, c13)],
                   dtype=complex)
    Uint = np.dot(U13, U12)
    U = np.dot(U23, Uint)

    UT = np.transpose(U)
    Udagger = np.conjugate(UT)

    # Mass basis Hamiltonian
    Hmass = np.array([(0, 0, 0), (0, mass2 / (2 * E_GeV), 0), (0, 0, mass3 / (2 * E_GeV))])

    # Flavor-basis Hamiltonian, without matter
    Hmid = np.dot(Hmass, Udagger)
    Hflav0 = np.dot(U, Hmid)

    # Full flavor-basis Hamiltonian. V_int represents the contribution to
    # potential due to charged-current neutrino-electron interactions.
    # Neutral current factors out of V_int and doesn't affect oscillations.
    density = density_function(x, y, z)
    vcc = density * 0.000193
    Vee = 1.0
    Vint = vcc * np.array([(Vee, 0, 0), (0, 0, 0), (0, 0, 0)], dtype=complex)
    Hflav = 5.1 * (Hflav0 + Vint)  # factor of 5.1 comes from units... see notes

    # Input array "amp" is four values: the real and imaginary components of
    # each flavor amplitude. Note that "j" instead of "i" is used.
    eAmp = amp[0] + 1.j * amp[1]
    muAmp = amp[2] + 1.j * amp[3]
    tauAmp = amp[4] + 1.j * amp[5]

    # This uses the (flavor-basis) Hamiltonian to define the time-evolution
    deAmpdx = -1.j * (Hflav[0, 0] * eAmp + Hflav[0, 1] * muAmp + Hflav[0, 2] * tauAmp)
    dmuAmpdx = -1.j * (Hflav[1, 0] * eAmp + Hflav[1, 1] * muAmp + Hflav[1, 2] * tauAmp)
    dtauAmpdx = -1.j * (Hflav[2, 0] * eAmp + Hflav[2, 1] * muAmp + Hflav[2, 2] * tauAmp)

    # Return an array of "d(Amplitude)/dx"
    dAmpdx = [np.real(deAmpdx), np.imag(deAmpdx), np.real(dmuAmpdx), np.imag(dmuAmpdx), np.real(dtauAmpdx),
              np.imag(dtauAmpdx)]
    return dAmpdx


def three_flavor_prob_any_moon_density(energy, y, z, density_function):  # energy in GeVs, y and z in km
    r_moon = 1737  # km
    x_final = np.sqrt(r_moon ** 2 - y ** 2 - z ** 2)
    x0 = -x_final

    x = np.linspace(x0, x_final, 1000)
    del13 = np.pi / 3
    solution = odeint(moon_deriv_three_flavor, turn_flavor_ratios_into_probability_amplitudes(2), x,
                      args=(energy, del13, y, z, density_function))
    ProbNuE = solution[:, 0] * solution[:, 0] + solution[:, 1] * solution[:, 1]
    ProbNuMu = solution[:, 2] * solution[:, 2] + solution[:, 3] * solution[:, 3]
    ProbNuTau = solution[:, 4] * solution[:, 4] + solution[:, 5] * solution[:, 5]
    return x, ProbNuE, ProbNuMu, ProbNuTau, solution


# uses monte carlo to simulate neutrinos of a certain energy passing through moon and travelling to earth
# then returns the expected amount of electron, muon, and tau neutrinos respectively as a fraction of 1 (e.g. [.25, .25, .5] means a quarter are each electron and muon and then half will be tau)
def monte_carlo_uniform_core(energy, core_mantle_density_ratio, core_radius):  # energy in GeVs
    r_moon = 1737  # km

    density_function = make_density_function_from_ratio(core_mantle_density_ratio, core_radius)

    e_prob = 0
    m_prob = 0
    t_prob = 0
    scale_factor = 0

    radii = np.linspace(0, r_moon, 1000)
    for radius in radii:
        after_moon = three_flavor_prob_any_moon_density(energy, radius, 0, density_function)
        e_prob += after_moon[1][-1] * radius
        m_prob += after_moon[2][-1] * radius
        t_prob += after_moon[3][-1] * radius
        scale_factor += radius

    # normalize probabilities
    e_prob /= scale_factor
    m_prob /= scale_factor
    t_prob /= scale_factor

    # now need to propogate to earth
    at_earth_ratios = decoherence_travel_earth(e_prob, m_prob, t_prob)
    return at_earth_ratios

"""

Neutrino flux estimation

"""

#This function returns cosmic ray flux per Gaisser
#units are /SR/s/M^2/log(Gev)
def get_cosmic_ray_flux(log_energy): # / log gev
    return 1.7 * (10 ** 4) * ((10 ** log_energy) ** -2.7) * (10 ** log_energy) * np.log(10)

#Determines how much energy a cosmic ray "loses" when it decays into a neutrino in our atmosphere
#current model is gaussian distribution centered around 1 log(GeV) with a standard deviation of 1/4 (log(Cosmic Ray Energy))
def decide_energy_loss(log_energy):
    energy_loss = np.random.normal(1, abs(log_energy) / 4, 1)[0]
    return energy_loss


# Uses a monte carlo simulation to create and a function which takes log neutrino energy as an input and returns differential neutrino flux
# Differential neutrino flux is in units of neutrinos/log(GeV)
def make_neutrino_flux_function_using_mc():
    # range of cosmic ray energies we care about
    log_min_cr_energy = -1
    log_max_cr_energy = 2

    # range of possible cosmic ray fluxes
    min_flux = get_cosmic_ray_flux(log_min_cr_energy)
    max_flux = get_cosmic_ray_flux(log_max_cr_energy)

    # make histogram to store neutrino counts
    # energy bins in log(GeV)
    log_neutrino_min_energy = log_min_cr_energy - 1.5
    log_neutrino_max_energy = log_max_cr_energy - 0.5
    num_bins = 100
    log_neutrino_energy_bins = np.linspace(log_neutrino_min_energy, log_neutrino_max_energy, num_bins + 1)
    neutrino_energy_histogram = np.zeros(num_bins)

    for i in range(10 ** 8):  # change back to 10 ** 8
        # pick random cosmic ray energy and associated CR flux
        log_random_CR_energy = random.uniform(log_min_cr_energy, log_max_cr_energy)
        random_flux = random.uniform(min_flux, max_flux)

        if random_flux < get_cosmic_ray_flux(
                log_random_CR_energy):  # decide whether to include cosmic ray (and the neutrino it produces)
            log_neutrino_energy = log_random_CR_energy - decide_energy_loss(
                log_random_CR_energy)  # decide what the produced neutrino's energy is
            bin_index = np.digitize(log_neutrino_energy, log_neutrino_energy_bins) - 1
            if 0 <= bin_index < num_bins:
                neutrino_energy_histogram[bin_index] += 1  # add neutrino to histogram

    bin_centers = (log_neutrino_energy_bins[:-1] + log_neutrino_energy_bins[1:]) / 2

    neutrino_count_interp = interp1d(bin_centers, neutrino_energy_histogram,
                                     kind='linear')  # interpolating neutrino energy histogram

    # We need out how many cosmic rays actually produced neutrinos in this energy range
    # So we calculate this, and then adjust our histogram interpolation to have the correct flux magnitude (and not just shape)
    # Assumes a 1-1 ratio of cosmic rays to neutrinos
    total_count = quad(neutrino_count_interp, log_min_cr_energy - 1.45, log_max_cr_energy - 0.55, limit=1000)[
        0]  # not ideal that this gives integration errors but I think it's alright
    total_cosmic_rays = quad(get_cosmic_ray_flux, log_min_cr_energy, log_max_cr_energy)[0]
    scale_factor = total_cosmic_rays / total_count

    #def neutrino_flux_interp(log_energy):
        #return neutrino_count_interp(log_energy) * scale_factor

    neutrino_flux_interp = interp1d(
        bin_centers,
        neutrino_energy_histogram * scale_factor,
        kind='linear')

    return neutrino_flux_interp

#This is all code I got from IceCube's website
#Allows me to get effective of IceCube (with upgrade)

# Define path to file (you may need to change this to match your system)
input_file = "neutrino_mc.csv"

# Load the file using pandas
input_data = pd.read_csv(input_file)

# Defining a few useful thing sbefore we get started...

# Define some energy bins (used throughout this notebook)
energy_bins_fine = np.logspace(0., 2., num=21)
energy_bins_course = np.logspace(0., 2., num=11)

# Define masks to identify different neutrino flavors
nue_mask = (np.abs(input_data["pdg"]) == 12)
numu_mask = (np.abs(input_data["pdg"]) == 14)
nutau_mask = (np.abs(input_data["pdg"]) == 16)

# It is often desirable to seperate the neutrinos by flavor and interaction.
# It is common to consider all Neutral Current (NC) interactions together, as
# there is no final state charged lepton with which to discriminate the different
# neutrino flavors.
# Define masks to identify different flavor/interaction combinations.
nc_mask = input_data["current_type"] == 0
cc_mask = input_data["current_type"] == 1
nue_cc_mask = nue_mask & cc_mask
numu_cc_mask = numu_mask & cc_mask
nutau_cc_mask = nutau_mask & cc_mask
# Choosing nue CC events as an example
chosen_mask = numu_cc_mask

# Calc effective area
effective_area_hist, bin_edges = np.histogram(input_data["true_energy"][chosen_mask], weights=input_data["weight"][chosen_mask], bins=energy_bins_fine )
effective_area_hist /= 4. * np.pi # Normalise by solid angle (using the fully sky)
effective_area_hist /= np.diff(bin_edges) # Bin widths

def get_aeff_upgrade(log_energy): #returns effective are of the updated ice cube in m^2
    bin_index = np.searchsorted(bin_edges, 10 ** log_energy, side='right')
    return effective_area_hist[bin_index]


# the upgrade aeff graph we get from IceCube is a step function
# this returns a smoother interpolation so that graphs don't come out looking strange
def make_aeff_upgrade_interp():
    log_aeff_energies = np.linspace(-3, 1.9, 30)
    aeffs = []

    for log_energy in log_aeff_energies:
        aeffs.append(get_aeff_upgrade(log_energy))

    aeff_interpolation = interp1d(log_aeff_energies, aeffs, kind="linear")
    return aeff_interpolation


# returns an estimate of differential interactions of cosmic ray neutrinos that pass through the moon and hit iceCube (upgraded)
# units are interactions/log(GeV)

#note you need to use get make_neutrino_flux_function_using_mc() and make_aeff_upgrade_interp() before you run this function

def get_diff_interactions(log_neu_energy):
    try:
        with open('diff-flux-estimate-functions.pkl', 'rb') as f:
            diff_flux_splines_catalog = pickle.load(f)
    except:
        diff_flux_splines_catalog = dict()
        diff_flux_splines_catalog['get_neutrino_flux'] = make_neutrino_flux_function_using_mc()
        diff_flux_splines_catalog['get_aeff_upgrade_interpolated'] = make_aeff_upgrade_interp()
        with open('diff-flux-estimate-functions.pkl', 'wb') as f:
            pickle.dump(diff_flux_splines_catalog, f)

    get_neutrino_flux = diff_flux_splines_catalog['get_neutrino_flux']
    get_aeff_upgrade_interpolated = diff_flux_splines_catalog['get_aeff_upgrade_interpolated']

    r_moon = 1737 * 10 ** 3  # m
    dist_moon_to_earth = 384400 * 10 ** 3  # m

    moon_surface_area = 4 * np.pi * (r_moon ** 2)  # m^2

    total_neutrino_flux = get_neutrino_flux(log_neu_energy) * moon_surface_area * 2 * np.pi  # total neutrinos coming from moon in all directions

    IceCube_flux_fraction = get_aeff_upgrade_interpolated(log_neu_energy) / (4 * np.pi * (
                dist_moon_to_earth ** 2))  # calculate what fraction total flux emitted would hit iceCube's effective area

    diff_interactions = total_neutrino_flux * IceCube_flux_fraction  # calculate the precise number if differential interactions

    return diff_interactions  # per log GeV


"""
    Mock Data Analysis
"""

def make_gaussian_pdf(mu, sigma, xmin, xmax):
    x = np.linspace(xmin, xmax, 1000)
    y = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma)**2
    )

    return interp1d(
        x, y,
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )


# creats a pdf of expected lunar neutrinos of muon-flavor detected between 1 and 10 GeVs
# is in terms of log/Gevs
# core_radius input is in km

#This is Prof. Hyde's code for making/saving PDFs, repurposed for muon detection pdfs
# import any previous energy pdf splines I've made.
# If file doesn't exist, create it and populate it with starter pdfs

try:
    with open('gaussian-pdf-splines-catalog.pkl', 'rb') as f:
        gaussian_pdf_splines_catalog = pickle.load(f)
except:
    gaussian_pdf_splines_catalog = dict()
    #this is slow but should only happen once

    starter_pdf = make_gaussian_pdf(0, 1, -100, 100)
    gaussian_pdf_splines_catalog['mu'] = [0]
    gaussian_pdf_splines_catalog['sigma'] = [1]
    gaussian_pdf_splines_catalog['spline'] = [starter_pdf]
    with open('gaussian-pdf-splines-catalog.pkl', 'wb') as f:
            pickle.dump(gaussian_pdf_splines_catalog, f)


# Retrieve appropriate muon pdf spline, or if it's not in the
# catalog, evaluate and add it

def gaussian_pdf_spline(find_mu, find_sigma):
    index_val = len(gaussian_pdf_splines_catalog['mu'])
    found = False
    for i in range(0, index_val):
        # round to 4th decimal place - avoid unnecessary replication of splines
        # due to rounding error in late decimal places
        c1 = (np.round(gaussian_pdf_splines_catalog['mu'][i], 2) == np.round(
            find_mu, 2))
        c2 = (np.round(gaussian_pdf_splines_catalog['sigma'][i], 0) == np.round(find_sigma, 0))
        if c1 and c2:
            found_spline = gaussian_pdf_splines_catalog['spline'][i]
            found = True

    if found == False:
        # generate new spline...
        print(r'Generating new energy pdf spline: mu =',
              np.round(find_mu, 2), r'sigma =', np.round(find_sigma, 0))
        this_spline = make_gaussian_pdf(find_mu, find_sigma)
        # ... and add to catalog...
        gaussian_pdf_splines_catalog['spline'].append(this_spline)
        gaussian_pdf_splines_catalog['mu'].append(np.round(find_mu, 2))
        gaussian_pdf_splines_catalog['sigma'].append(np.round(find_sigma, 0))
        found_spline = this_spline
        # ... and save to file
        with open('gaussian-pdf-splines-catalog.pkl', 'wb') as f:
            pickle.dump(gaussian_pdf_splines_catalog, f)

    return found_spline


# Wrapper for energy pdf splines: calling this function evaluates
# the signal energy pdf value for each point in sample

def eval_gaussian_pdf(sample, mu, sigma):
    muon_pdf = gaussian_pdf_spline(mu, sigma)
    muon_pdf_on_sample = muon_pdf(sample)
    return muon_pdf_on_sample


# creates mock muon neutrino detection data
# returns a list of log neutrino energies
def make_mock_data(mu, sigma, num_neutrinos):
    pdf = gaussian_pdf_spline(mu,
                          sigma)  # make_muon_detections_pdf(core_mantle_density_ratio, core_radius)

    log_energies = []
    while len(log_energies) < num_neutrinos:
        random_log_energy = random.uniform(-1, 1)
        random_probability = random.uniform(0, 3)

        if random_probability < pdf(random_log_energy):
            log_energies.append(random_log_energy)

    return log_energies


# finds log likelihood of data for a given model (density ratio and radius) using muon neutrino pdfs
# data is a list of log neutrino energies
def find_log_likelihood(mu, sigma, data):
    muon_pdf = gaussian_pdf_spline(mu, sigma)
    log_likelihood = 0
    for i in range(len(data)):
        log_likelihood += np.log10(muon_pdf(data[i]))

    return log_likelihood


# this function finds the test statistic for a given model (density ratio and radius) and data set
# test statistic is defined as lambda = 2 log([likelihood of test hypothesis] / [likelihood of null hypothesis])
# null hypothesis is radius = 330 km, core-mantle density ratio = 1.77 (core density is thus 6.15 g/cm^2)
# data is a list of log neutrino energies

def find_test_statistic(mu_test_hypothesis, sigma_test_hypothesis, data):
    test_hypothesis_log_likelihood = find_log_likelihood(mu_test_hypothesis,
                                                         sigma_test_hypothesis, data)
    null_hypothesis_log_likelihood = find_log_likelihood(0, 1, data)

    test_statistic = 2 * (test_hypothesis_log_likelihood - null_hypothesis_log_likelihood)
    return test_statistic


# takes in a list of log detected neutrino energies
# then returns the best fit density ratio and radius by maximizing test statistic across a range of parameters
# data is a list of log neutrino energies
def find_best_fit_parameters(data):
    mus = np.linspace(-10, 10, 10) #eventually change to 100
    sigmas = np.linspace(1, 10, 10)  # in km

    first_test = True  # whether or not we have tested a pair of parameters yet
    best_fit_test_statistic = 0
    best_fit_density_ratio = 0
    best_fit_radius = 0

    for core_mantle_density_ratio in mus:
        for core_radius in sigmas:
            if first_test:  # if first set of parameters, they are automatically the best
                best_fit_density_ratio = core_mantle_density_ratio
                best_fit_radius = core_radius
                best_fit_test_statistic = find_test_statistic(core_mantle_density_ratio, core_radius, data)
                first_test = False
            else:
                new_test_statistic = find_test_statistic(core_mantle_density_ratio, core_radius, data)
                if (new_test_statistic > best_fit_test_statistic):
                    best_fit_density_ratio = core_mantle_density_ratio
                    best_fit_radius = core_radius

    return best_fit_density_ratio, best_fit_radius, best_fit_test_statistic


# create many null hypothesis mock data sets for find_confidence_interval function
# returns a list of mock data sets of log neutrino energies
def make_mock_data_sets(mu, sigma, num_mock_data_sets, neutrinos_per_set):
    data_sets = []

    for i in range(num_mock_data_sets):
        data_sets.append(make_mock_data(mu, sigma,
                                        neutrinos_per_set))

    return data_sets


# given a hypothesis's test statistic, finds and returns how significant that result is (the p value)
# i.e. how often would the null hypothesis data set produce such a high test statistic
def find_significance(hypothesis_test_statistic, num_mock_data_sets, neutrinos_per_data_set):
    null_hypothesis_data_sets = make_mock_data_sets(0, 1, num_mock_data_sets, neutrinos_per_data_set)

    num_significant_data_sets = 0.0  # this is to keep track of the number of mock data sets with a higher test statistic than our test hypothesis

    for data_set in null_hypothesis_data_sets:
        best_fit_test_statistic = find_best_fit_parameters(data_set)[2]
        if best_fit_test_statistic > hypothesis_test_statistic:  # does this data set produce a higher test statistic?
            num_significant_data_sets += 1

    return num_significant_data_sets / num_mock_data_sets


# given a test hypothesis ratio and radius, calculates the minimum test statistic to consider for a given confidence interval
def find_confidence_interval_minimum_test_statistics(percentiles, test_hypothesis_mu, test_hypothesis_sigma,
                                                    num_mock_data_sets, neutrinos_per_data_set):
    test_hypothesis_data_sets = make_mock_data_sets(test_hypothesis_mu, test_hypothesis_sigma,
                                                    num_mock_data_sets, neutrinos_per_data_set)

    best_fit_mus = []
    best_fit_sigmas = []
    best_fit_test_statistics = []

    for data_set in test_hypothesis_data_sets:
        new_density_ratio, new_radius, new_test_statistic = find_best_fit_parameters(
            data_set)  # am i finding test statistic correctly???
        best_fit_mus.append(new_density_ratio)
        best_fit_sigmas.append(new_radius)
        best_fit_test_statistics.append(new_test_statistic)

    minimum_test_statistics = []

    for percentile in percentiles:
        minimum_test_statistics.append(np.percentile(best_fit_test_statistics, 100 - percentile))


    return tuple(minimum_test_statistics)



# makes a confidence interval plot, but instead creates three contours (at 30, 60 and 90 pct confidence)
# this is overlayed on a heatmap showing the test statistic of each parameter combination
def make_heatmap_confidence_interval_plot(real_ratio, real_radius, num_neutrinos, percentiles=(30,60,90), save_figure=True):
    # CHANGE FONTSIZE (fontsize=12)
    #first sets up the data
    real_data = make_mock_data(real_ratio, real_radius, num_neutrinos)
    core_mantle_density_ratios = np.linspace(1, 15, 10)
    core_radii = np.linspace(200, 600, 10)  # in km

    ratio_radius_test_statistic_tuples = []  # list of tuples with corresponding ratio, radius and test statistic

    for core_mantle_density_ratio in core_mantle_density_ratios:
        for core_radius in core_radii:
            new_test_statistic = find_test_statistic(core_mantle_density_ratio, core_radius, real_data)
            ratio_radius_test_statistic_tuples.append((core_mantle_density_ratio, core_radius, new_test_statistic))

    ratios, radii, test_statistics = zip(*ratio_radius_test_statistic_tuples)

    best_fit_density_ratio, best_fit_radius, best_fit_test_statistic = find_best_fit_parameters(real_data)

    smallest_pct_min_stat, middle_pct_min_stat, biggest_pct_min_stat = find_confidence_interval_minimum_test_statistics(percentiles,
                                                                                                 best_fit_density_ratio,
                                                                                                 best_fit_radius,
                                                                                                 1000, num_neutrinos)


    #now plotting the data

    max_test_statistic = max(test_statistics)
    delta_test_statistics = []

    for test_statistic in test_statistics:
        delta_test_statistics.append(test_statistic - max_test_statistic)

    # Step 1: Create a 2D grid
    ratios = np.array(ratios)
    radii = np.array(radii)
    delta_stats = np.array(delta_test_statistics)

    # Get unique sorted values
    unique_ratios = np.unique(ratios)
    unique_radii = np.unique(radii)

    # Create a meshgrid
    X, Y = np.meshgrid(unique_ratios, unique_radii)

    # Step 2: Reshape stats into a 2D array Z
    Z = np.full(X.shape, np.nan)
    for i in range(len(delta_stats)):
        x_idx = np.where(unique_ratios == ratios[i])[0][0]
        y_idx = np.where(unique_radii == radii[i])[0][0]
        Z[y_idx, x_idx] = delta_stats[i]

    # Step 3: Plot heatmap
    plt.figure(figsize=(10, 8))
    heatmap = plt.pcolormesh(X, Y, Z, shading='auto', cmap='Spectral')
    plt.colorbar(heatmap, label=r'$\Delta_{\lambda}$')

    # Step 4: Add contours
    ratio_points = np.linspace(min(ratios), max(ratios), 100)
    radius_points = np.linspace(min(radii), max(radii), 100)
    ratio_grid, radii_grid = np.meshgrid(ratio_points, radius_points)

    test_statistics_grid = griddata((ratios, radii), test_statistics, (ratio_grid, radii_grid), method='linear')

    linestyles_dict = {
        -100: '--',
        biggest_pct_min_stat: '--',
        middle_pct_min_stat: '-.',
        smallest_pct_min_stat: ':',
        100: '-'
    }

    # contour levels must be increasing
    if (middle_pct_min_stat <= biggest_pct_min_stat) or (smallest_pct_min_stat <= middle_pct_min_stat):
        print("error: contour levels werent increasing")
        print("tried to make" + "heatmap_confidence_interval_plot_ratio" + str(real_ratio) + "_radius" + str(
            real_radius) + "_neutrinos" + str(num_neutrinos))
        return (1)

    contours = plt.contour(ratio_grid, radii_grid, test_statistics_grid,
                           levels=[-100, biggest_pct_min_stat, middle_pct_min_stat, smallest_pct_min_stat, 100],
                           colors=['k', 'k'],
                           linestyles=["--", "--", "-.", ":", "--"])

    plt.clabel(contours, inline=True, fontsize=8)

    # finding best fit parameters
    max_test_statistic_index = test_statistics.index(max_test_statistic)
    best_fit_ratio = ratios[max_test_statistic_index]
    best_fit_radius = radii[max_test_statistic_index]

    plt.scatter(best_fit_ratio, best_fit_radius, marker='x', color='black', s=100)
    plt.annotate('Best Fit Value',
                 (best_fit_ratio, best_fit_radius),
                 textcoords="offset points",
                 xytext=(10, 10),  # offset for text placement
                 ha='left',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

    plt.scatter(real_ratio, real_radius, color='black', s=100, edgecolor='black', label="True Values")
    plt.annotate('\"True\" Value',
                 (real_ratio, real_radius),
                 textcoords="offset points",
                 xytext=(10, 10),  # offset for text placement
                 ha='left',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color='black', linestyle=':', label=str(percentiles[0]) + '% confidence'),
        Line2D([0], [0], color='black', linestyle='-.', label=str(percentiles[1]) +'% confidence'),
        Line2D([0], [0], color='black', linestyle='--', label=str(percentiles[2]) +'% confidence'),
    ]

    # Add manual legend
    plt.legend(handles=custom_lines, loc='best')

    # Final touches
    plt.xlabel(r"$\mu}$(True = " + str(real_ratio) + ")")
    plt.ylabel(r"$\sigma$ (km) " + "(True = " + str(real_radius) + ")")
    plt.title("Heatmap Confidence Interval Plot Gaussian")
    plt.annotate('num neutrinos = ' + str(num_neutrinos), xy=(1.00, 1.02), xycoords='axes fraction', ha='center')
    plt.annotate('\"True\" Value',
                 (real_ratio, real_radius),
                 textcoords="offset points",
                 xytext=(10, 10),  # offset for text placement
                 ha='left',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

    if save_figure:
        plt.savefig(
            "gaussian_heatmap_confidence_interval_plot_mu" + str(real_ratio) + "_sigma" + str(real_radius) + "_neutrinos" + str(
                num_neutrinos) + "_contours" + "_".join(map(str, percentiles)) + ".png")
    #plt.show()


def make_scatter_plot(real_ratio, real_radius, num_neutrinos, num_scatters):
    plt.figure(figsize=(10, 8))
    for i in range(num_scatters):
        mock_data = make_mock_data(real_ratio, real_radius, num_neutrinos)
        best_fit_ratio, best_fit_radius, test_statistic = find_best_fit_parameters(mock_data)
        plt.scatter(best_fit_ratio, best_fit_radius, marker='x', color='black', s=100)

    plt.xlabel(r"$\rho_{core} / \rho_{mantle}$(True = " + str(real_ratio) + ")")
    plt.ylabel(r"$R_{core}$ (km) " + "(True = " + str(real_radius) + ")")
    plt.title("Best Fit Scatter Plot")
    plt.annotate('num neutrinos = ' + str(num_neutrinos), xy=(1.00, 1.02), xycoords='axes fraction', ha='center')
    plt.scatter(real_ratio, real_radius, marker='x', color='red', s=100)
    plt.annotate('\"True\" Value',
                 (real_ratio, real_radius),
                 textcoords="offset points",
                 xytext=(10, 10),  # offset for text placement
                 ha='left',
                 fontsize=10,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig(
        "best_fit_scatter_plot_ratio" + str(real_ratio) + "_radius" + str(
            real_radius) + "_neutrinos" + str(
            num_neutrinos) + "_scatters" + str(num_scatters) + ".png")


