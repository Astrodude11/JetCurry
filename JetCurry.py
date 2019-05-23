# Copyright 2017. Katie Kosak. Eric Perlman
#    This file is part of JetCurry.
#    JetCurry is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math
from pylab import *
from matplotlib import *
from scipy.interpolate import spline
import emcee
from scipy.optimize import fmin_l_bfgs_b
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")

np.seterr(all='ignore')

def imagesqrt(image, scale_min, scale_max):
    '''
    Algorithm Courtesy of Min-Su Shin (msshin @ umich.edu)
    Modified by Katie Kosak 07/02/2015 to fit the needs of Hubble Data for
    Dr. Perlman and Dr. Avachat.

    Arguments:
        image : numpy array
            FITS image
        scale_min : float
            Minimum scale
        scale_max : float
            Maximum scale

    Returns:
        imageData : numpy array
            Square root scale of image

    '''
    imageData = np.array(image, copy=True)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    imageData = np.sqrt(imageData)
    imageData = imageData / math.sqrt(scale_max - scale_min)
    return imageData


def Find_MaxFlux(file1, Upstream_Bounds, Downstream_Bounds, number_of_points):
    '''
    Calculates the max along each column

    Arguments:
        file1 : numpy array
            Sqrt image of FITS file
        Upstream_Bounds : numpy array
            Start position of jet
        Downstream_Bounds : numpy array
            End position of jet
        number_of_points : numpy integer
            Number of pixels along x-axis of the start and end positions of jet

    Returns:
        intensity_xpos : numpy array
            Max intensity at point x
        intensity_ypos : numpy array
            Max intensity at point y
        x_smooth : numpy array
            Returns number_of_points evenly spaced samples
                calculated over the start/stop interval
        y_smooth : numpy array
            Interpolate a curve using spline fit
        intensity_max : numpy array
            Max intensity of each column
    '''
    height, width = file1.shape
    intensity_max = np.array([])
    intensity_ypos = np.array([])
    intensity_xpos = np.array([])
    temp = np.zeros(shape=(height, 1))
    for k in range(Upstream_Bounds[0], Downstream_Bounds[0] - 1):
        for j in range(0, height - 1):
            pixel = file1[j, k]
            temp[j] = pixel
        pixel_max = np.nanmax(temp)
        position = [i for i, j in enumerate(temp) if j == pixel_max]
        intensity_max = np.append(intensity_max, pixel_max)
        intensity_ypos = np.append(intensity_ypos, position[0])
        intensity_xpos = np.append(intensity_xpos, k)
    x_smooth = np.linspace(
        Upstream_Bounds[0],
        Downstream_Bounds[0],
        num=number_of_points)
    y_smooth = spline(intensity_xpos, intensity_ypos, x_smooth)
    return intensity_xpos, intensity_ypos, x_smooth, y_smooth, intensity_max


def Calculate_s_and_eta(x_smooth, y_smooth, core_points, output_directory, filename):
    '''
    Calculates S and ETA values

    Arguments:
        x_smooth : numpy array
            Evenly spaced samples
        y_smooth : numpy array
            Interpolate a curve using spline fit
        core_points : numpy array
            Start position of jet
        output_directory : string
            Path of location to save data products
        filename : string
            Root name of file to be saved

    Returns:
        s: list
        eta: list
    '''
    s = []
    eta = []

    for i in range(len(x_smooth)):
        x = x_smooth[i] - float(core_points[1])  # default core_points[1]
        y = y_smooth[i] - float(core_points[0])  # default core_points[0]
        s_value = (x**2 + y**2)**(0.5)
        eta_value = math.atan(y / x)
        s.append(s_value)
        eta.append(eta_value)

    with open(output_directory + filename + '_parameters.txt', 'w') as file:
        for x in range(len(x_smooth)):
            file.write(str(s[x]) + '\t' + str(eta[x]) + '\t' +
                       str(x_smooth[x]) + '\t' + str(y_smooth[x]) + '\n')
    return s, eta


def Run_MCMC1(s, eta, theta, ind, output_directory, filename):
    '''
    '''
    ndim, nwalkers, nsteps = 5, 1024, 50
    initial_alpha = np.arange(0, 2.0, 0.5)
    initial_beta = np.arange(0, 2.0, 0.5)
    initial_phi = np.arange(0, 2.0, 0.5)
    initial_xi = np.arange(0, 2.0, 0.5)
    initial_d = np.arange(floor(s), floor(s) + 4 * 20.25, 20.25)
    pos = []
    for i in range(len(initial_alpha)):
        for j in range(len(initial_beta)):
            for k in range(len(initial_phi)):
                for l in range(len(initial_xi)):
                    for m in range(len(initial_d)):
                        init = []
                        init.append(initial_alpha[i])
                        init.append(initial_beta[j])
                        init.append(initial_phi[k])
                        init.append(initial_xi[l])
                        init.append(initial_d[m])
                        pos.append(init)

    def lnlike(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        model = (((s * np.sin(eta)) / np.sin(phi))**2 + (s * np.cos(eta) * (np.sin(theta) * np.cos(alpha) + np.sin(alpha) * np.cos(theta)) / np.cos(alpha))**2 - d**2)**2 + (((np.sin(beta) * np.cos(alpha)) / (np.sin(alpha) * np.cos(beta)))**2 - (np.cos(eta)**2))**2 + (d * np.cos(xi) * np.cos(theta) - ((s * np.cos(eta) * np.sin(alpha)) / np.cos(alpha)) - d * np.sin(xi) * np.cos(phi) * np.sin(theta))**2 + ((np.sin(eta) / np.cos(eta)) - ((np.sin(xi) * np.sin(phi)) / (np.cos(xi) * np.sin(theta) + np.sin(xi) * np.cos(phi) * np.cos(theta))))**2 + (s - d * np.cos(beta))**2
        return -np.log(np.abs(model) + 1)

    t = 1.5708 - theta

    def lnprior(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        if 0 < alpha < 1.57 and 0 < beta < 1.57 and 0 < phi < 3.14 and 0 < xi < t and floor(s) < d < floor(s) + 3 * 20.25:
            return 0.0
        return np.inf

    def lnprob(v):
        '''
        '''
        lp = lnprior(v)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(v)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    probs = sampler.flatlnprobability
    A = np.array(probs)
    maximum_indices = np.where(A == max(probs))
    pvectors = []
    pvectors = samples[maximum_indices]
    r = pvectors[0]

    with open(output_directory + filename + '_MCMC1.txt', 'a') as file:
        file.write(str(r[0]) +
                   '\t' +
                   str(r[1]) +
                   '\t' +
                   str(r[2]) +
                   '\t' +
                   str(r[3]) +
                   '\t' +
                   str(r[4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def MCMC1_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    for i in range(0, int(len(s) - 3.0), 4):
        r = Process(target=Run_MCMC1, args=(s[i], eta[i], theta, i, output_directory, filename))
        r.start()
        r.join()

        r1 = Process(target=Run_MCMC1, args=(
            s[i + 1], eta[i + 1], theta, i + 1, output_directory, filename))
        r1.start()
        r1.join()

        r2 = Process(target=Run_MCMC1, args=(
            s[i + 2], eta[i + 2], theta, i + 2, output_directory, filename))
        r2.start()
        r2.join()

        r3 = Process(target=Run_MCMC1, args=(
            s[i + 3], eta[i + 3], theta, i + 3, output_directory, filename))
        r3.start()
        r3.join()

    return


def RunMCMC2(s, eta, d0, theta, a, b, c, e, ind, output_directory, filename):
    '''
    '''
    initial_alpha = np.arange(a, a + 0.2, 0.05)
    initial_beta = np.arange(b, b + 0.2, 0.05)
    initial_phi = np.arange(c, c + 0.2, 0.05)
    initial_xi = np.arange(e, e + 0.2, 0.05)
    initial_d = np.arange(floor(d0), floor(d0) + 2.0, 0.5)

    pos = []
    for i in range(len(initial_alpha)):
        for j in range(len(initial_beta)):
            for k in range(len(initial_phi)):
                for l in range(len(initial_xi)):
                    for m in range(len(initial_d)):
                        init = []
                        init.append(initial_alpha[i])
                        init.append(initial_beta[j])
                        init.append(initial_phi[k])
                        init.append(initial_xi[l])
                        init.append(initial_d[m])
                        pos.append(init)

    ndim, nwalkers, nsteps = 5, int(shape(pos)[0]), 50

    def lnlike(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        model = (((s * np.sin(eta)) / np.sin(phi))**2 + (s * np.cos(eta) * (np.sin(theta) * np.cos(alpha) + np.sin(alpha) * np.cos(theta)) / np.cos(alpha))**2 - d**2)**2 + (((np.sin(beta) * np.cos(alpha)) / (np.sin(alpha) * np.cos(beta)))**2 - (np.cos(eta)**2))**2 + (d * np.cos(xi) * np.cos(theta) - ((s * np.cos(eta) * np.sin(alpha)) / np.cos(alpha)) - d * np.sin(xi) * np.cos(phi) * np.sin(theta))**2 + ((np.sin(eta) / np.cos(eta)) - ((np.sin(xi) * np.sin(phi)) / (np.cos(xi) * np.sin(theta) + np.sin(xi) * np.cos(phi) * np.cos(theta))))**2 + (s - d * np.cos(beta))**2
        return -np.log(np.abs(model) + 1)

    t = 1.5708 - theta

    def lnprior(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        if a < alpha < (
                a +
                0.2) and b < beta < (
                b +
                0.2) and c < phi < (
                c +
                0.2) and e < xi < (
                    e +
                    0.2) and floor(d0) < d < (
                        floor(d0) +
                        3 *
                20.25):
            return 0.0
        return np.inf

    def lnprob(v):
        '''
        '''
        lp = lnprior(v)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(v)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    probs = sampler.flatlnprobability
    A = np.array(probs)
    maximum_indices = np.where(A == max(probs))
    pvectors = []
    pvectors = samples[maximum_indices]
    h = pvectors[0]

    with open(output_directory + filename + '_MCMC2.txt', 'a') as file:
        file.write(str(h[0]) +
                   '\t' +
                   str(h[1]) +
                   '\t' +
                   str(h[2]) +
                   '\t' +
                   str(h[3]) +
                   '\t' +
                   str(h[4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def MCMC2_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_first = []
    beta_first = []
    phi_first = []
    xi_first = []
    d_first = []
    for line in open(output_directory + filename + '_MCMC1.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        # if line.startswith('\n'):
        #    continue
        fields = line.split()
        alpha_first.append(fields[0])
        beta_first.append(fields[1])
        phi_first.append(fields[2])
        xi_first.append(fields[3])
        d_first.append(fields[4])
    alpha_first = [float(i) for i in alpha_first]
    beta_first = [float(i) for i in beta_first]
    phi_first = [float(i) for i in phi_first]
    xi_first = [float(i) for i in xi_first]
    d_first = [float(i) for i in d_first]
    for j in range(0, len(alpha_first), 4):
        alpha0 = float(alpha_first[j])
        beta0 = float(beta_first[j])
        phi0 = float(phi_first[j])
        xi0 = float(xi_first[j])
        d0 = float(d_first[j])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        h = Process(
            target=RunMCMC2,
            args=(
                s[j],
                eta[j],
                d0,
                theta,
                a,
                b,
                c,
                e,
                j, output_directory, filename))
        h.start()
        h.join()
        alpha1 = float(alpha_first[j + 1])
        beta1 = float(beta_first[j + 1])
        phi1 = float(phi_first[j + 1])
        xi1 = float(xi_first[j + 1])
        d1 = float(d_first[j + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        h1 = Process(target=RunMCMC2, args=(
            s[j + 1], eta[j + 1], d1, theta, a1, b1, c1, e1, j + 1, output_directory, filename))
        h1.start()
        h1.join()
        alpha2 = float(alpha_first[j + 2])
        beta2 = float(beta_first[j + 2])
        phi2 = float(phi_first[j + 2])
        xi2 = float(xi_first[j + 2])
        d2 = float(d_first[j + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        h2 = Process(target=RunMCMC2, args=(
            s[j + 2], eta[j + 2], d2, theta, a2, b2, c2, e2, j + 2, output_directory, filename))
        h2.start()
        h2.join()
        alpha3 = float(alpha_first[j + 3])
        beta3 = float(beta_first[j + 3])
        phi3 = float(phi_first[j + 3])
        xi3 = float(xi_first[j + 3])
        d3 = float(d_first[j + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        h3 = Process(target=RunMCMC2, args=(
            s[j + 3], eta[j + 3], d3, theta, a3, b3, c3, e3, j + 3, output_directory, filename))
        h3.start()
        h3.join()
    return

def Annealing1(eta, s, theta, alpha0, beta0, phi0, xi0, d0, a, b, c, e, ind, output_directory, filename):
    '''
    '''

    def f(x):
        '''
        '''
        alpha, beta, phi, xi, d = x
        return ((((s *
                   np.sin(eta)) /
                  np.sin(phi))**2 +
                 (s *
                  np.cos(eta) *
                  (np.sin(theta) *
                   np.cos(alpha) +
                   np.sin(alpha) *
                   np.cos(theta)) /
                  np.cos(alpha))**2 -
                 d**2)**2 +
                (((np.sin(beta) *
                   np.cos(alpha)) /
                  (np.sin(alpha) *
                   np.cos(beta)))**2 -
                 (np.cos(eta)**2))**2 +
                (d *
                 np.cos(xi) *
                 np.cos(theta) -
                 ((s *
                   np.cos(eta) *
                   np.sin(alpha)) /
                  np.cos(alpha)) -
                 d *
                 np.sin(xi) *
                 np.cos(phi) *
                 np.sin(theta))**2 +
                ((np.sin(eta) /
                  np.cos(eta)) -
                 ((np.sin(xi) *
                   np.sin(phi)) /
                    (np.cos(xi) *
                     np.sin(theta) +
                     np.sin(xi) *
                     np.cos(phi) *
                     np.cos(theta))))**2 +
                (s -
                 d *
                 np.cos(beta))**2)
    x0 = np.array([float(alpha0), float(beta0),
                   float(phi0), float(xi0), float(d0)])
    xmin = [a, b, c, e, float(floor(d0))]
    xmax = [float(a + 0.2), float(b + 0.2), float(c + 0.2),
            float(e + 0.2), float(floor(d0) + 2.0)]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    res = fmin_l_bfgs_b(
        f,
        x0,
        fprime=None,
        args=(),
        approx_grad=True,
        bounds=bounds,
        m=10,
        factr=10000000.0,
        pgtol=1e-05,
        epsilon=1e-08,
        iprint=-1,
        maxfun=150,
        maxiter=150,
        disp=None,
        callback=None)
    q = res

    with open(output_directory + filename + '_ANNE1.txt', 'a') as file:
        file.write(str(q[0][0]) +
                   '\t' +
                   str(q[0][1]) +
                   '\t' +
                   str(q[0][2]) +
                   '\t' +
                   str(q[0][3]) +
                   '\t' +
                   str(q[0][4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def Annealing1_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_MCMC2 = []
    beta_MCMC2 = []
    phi_MCMC2 = []
    xi_MCMC2 = []
    d_MCMC2 = []
    for line in open(output_directory + filename + '_MCMC2.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        object_alpha = fields[0]
        alpha_MCMC2.append(object_alpha)
        object_beta = fields[1]
        beta_MCMC2.append(object_beta)
        object_phi = fields[2]
        phi_MCMC2.append(object_phi)
        object_xi = fields[3]
        xi_MCMC2.append(object_xi)
        object_d = fields[4]
        d_MCMC2.append(object_d)
    for k in range(0, len(alpha_MCMC2), 4):
        alpha0 = float(alpha_MCMC2[k])
        beta0 = float(beta_MCMC2[k])
        phi0 = float(phi_MCMC2[k])
        xi0 = float(xi_MCMC2[k])
        d0 = float(d_MCMC2[k])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        q = Process(
            target=Annealing1,
            args=(
                eta[k],
                s[k],
                theta,
                alpha0,
                beta0,
                phi0,
                xi0,
                d0,
                a,
                b,
                c,
                e,
                k, output_directory, filename))
        q.start()
        q.join()
        alpha1 = float(alpha_MCMC2[k + 1])
        beta1 = float(beta_MCMC2[k + 1])
        phi1 = float(phi_MCMC2[k + 1])
        xi1 = float(xi_MCMC2[k + 1])
        d1 = float(d_MCMC2[k + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        q1 = Process(target=Annealing1,
                     args=(eta[k + 1],
                           s[k + 1],
                           theta,
                           alpha1,
                           beta1,
                           phi1,
                           xi1,
                           d1,
                           a1,
                           b1,
                           c1,
                           e1,
                           k + 1, output_directory, filename))
        q1.start()
        q1.join()
        alpha2 = float(alpha_MCMC2[k + 2])
        beta2 = float(beta_MCMC2[k + 2])
        phi2 = float(phi_MCMC2[k + 2])
        xi2 = float(xi_MCMC2[k + 2])
        d2 = float(d_MCMC2[k + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        q2 = Process(target=Annealing1,
                     args=(eta[k + 2],
                           s[k + 2],
                           theta,
                           alpha2,
                           beta2,
                           phi2,
                           xi2,
                           d2,
                           a2,
                           b2,
                           c2,
                           e2,
                           k + 2, output_directory, filename))
        q2.start()
        q2.join()
        alpha3 = float(alpha_MCMC2[k + 3])
        beta3 = float(beta_MCMC2[k + 3])
        phi3 = float(phi_MCMC2[k + 3])
        xi3 = float(xi_MCMC2[k + 3])
        d3 = float(d_MCMC2[k + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        q3 = Process(target=Annealing1,
                     args=(eta[k + 3],
                           s[k + 3],
                           theta,
                           alpha3,
                           beta3,
                           phi3,
                           xi3,
                           d3,
                           a3,
                           b3,
                           c3,
                           e3,
                           k + 3, output_directory, filename))
        q3.start()
        q3.join()
    return

def Annealing2(eta, s, theta, alpha0, beta0, phi0, xi0, d0, a, b, c, e, ind, output_directory, filename):
    '''
    '''

    def f(x):
        '''
        '''
        alpha, beta, phi, xi, d = x
        return ((((s *
                   np.sin(eta)) /
                  np.sin(phi))**2 +
                 (s *
                  np.cos(eta) *
                  (np.sin(theta) *
                   np.cos(alpha) +
                   np.sin(alpha) *
                   np.cos(theta)) /
                  np.cos(alpha))**2 -
                 d**2)**2 +
                (((np.sin(beta) *
                   np.cos(alpha)) /
                  (np.sin(alpha) *
                   np.cos(beta)))**2 -
                 (np.cos(eta)**2))**2 +
                (d *
                 np.cos(xi) *
                 np.cos(theta) -
                 ((s *
                   np.cos(eta) *
                   np.sin(alpha)) /
                  np.cos(alpha)) -
                 d *
                 np.sin(xi) *
                 np.cos(phi) *
                 np.sin(theta))**2 +
                ((np.sin(eta) /
                  np.cos(eta)) -
                 ((np.sin(xi) *
                   np.sin(phi)) /
                    (np.cos(xi) *
                     np.sin(theta) +
                     np.sin(xi) *
                     np.cos(phi) *
                     np.cos(theta))))**2 +
                (s -
                 d *
                 np.cos(beta))**2)
    x0 = np.array([float(alpha0), float(beta0),
                   float(phi0), float(xi0), float(d0)])
    xmin = [a, b, c, e, float(floor(d0))]
    xmax = [float(a + 0.1), float(b + 0.1), float(c + 0.1),
            float(e + 0.1), float(floor(d0) + 1.0)]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    res = fmin_l_bfgs_b(
        f,
        x0,
        fprime=None,
        args=(),
        approx_grad=True,
        bounds=bounds,
        m=10,
        factr=10000000.0,
        pgtol=1e-05,
        epsilon=1e-08,
        iprint=-1,
        maxfun=150,
        maxiter=150,
        disp=None,
        callback=None)
    q = res

    with open(output_directory + filename + '_ANNE2.txt', 'a') as file:
        file.write(str(q[0][0]) +
                   '\t' +
                   str(q[0][1]) +
                   '\t' +
                   str(q[0][2]) +
                   '\t' +
                   str(q[0][3]) +
                   '\t' +
                   str(q[0][4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return

def Annealing2_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_ANNE1 = []
    beta_ANNE1 = []
    phi_ANNE1 = []
    xi_ANNE1 = []
    d_ANNE1 = []

    for line in open(output_directory + filename + '_ANNE1.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        object_alpha = fields[0]
        alpha_ANNE1.append(object_alpha)
        object_beta = fields[1]
        beta_ANNE1.append(object_beta)
        object_phi = fields[2]
        phi_ANNE1.append(object_phi)
        object_xi = fields[3]
        xi_ANNE1.append(object_xi)
        object_d = fields[4]
        d_ANNE1.append(object_d)
    for k in range(0, len(alpha_ANNE1), 4):
        alpha0 = float(alpha_ANNE1[k])
        beta0 = float(beta_ANNE1[k])
        phi0 = float(phi_ANNE1[k])
        xi0 = float(xi_ANNE1[k])
        d0 = float(d_ANNE1[k])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        q = Process(
            target=Annealing2,
            args=(
                eta[k],
                s[k],
                theta,
                alpha0,
                beta0,
                phi0,
                xi0,
                d0,
                a,
                b,
                c,
                e,
                k, output_directory, filename))
        q.start()
        q.join()
        alpha1 = float(alpha_ANNE1[k + 1])
        beta1 = float(beta_ANNE1[k + 1])
        phi1 = float(phi_ANNE1[k + 1])
        xi1 = float(xi_ANNE1[k + 1])
        d1 = float(d_ANNE1[k + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        q1 = Process(target=Annealing2,
                     args=(eta[k + 1],
                           s[k + 1],
                           theta,
                           alpha1,
                           beta1,
                           phi1,
                           xi1,
                           d1,
                           a1,
                           b1,
                           c1,
                           e1,
                           k + 1, output_directory, filename))
        q1.start()
        q1.join()
        alpha2 = float(alpha_ANNE1[k + 2])
        beta2 = float(beta_ANNE1[k + 2])
        phi2 = float(phi_ANNE1[k + 2])
        xi2 = float(xi_ANNE1[k + 2])
        d2 = float(d_ANNE1[k + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        q2 = Process(target=Annealing2,
                     args=(eta[k + 2],
                           s[k + 2],
                           theta,
                           alpha2,
                           beta2,
                           phi2,
                           xi2,
                           d2,
                           a2,
                           b2,
                           c2,
                           e2,
                           k + 2, output_directory, filename))
        q2.start()
        q2.join()
        alpha3 = float(alpha_ANNE1[k + 3])
        beta3 = float(beta_ANNE1[k + 3])
        phi3 = float(phi_ANNE1[k + 3])
        xi3 = float(xi_ANNE1[k + 3])
        d3 = float(d_ANNE1[k + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        q3 = Process(target=Annealing2,
                     args=(eta[k + 3],
                           s[k + 3],
                           theta,
                           alpha3,
                           beta3,
                           phi3,
                           xi3,
                           d3,
                           a3,
                           b3,
                           c3,
                           e3,
                           k + 3, output_directory, filename))
        q3.start()
        q3.join()
    return


def Convert_Results_Cartesian(s, eta, theta, output_directory, filename):
    '''
    '''
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    alpha = []
    beta = []
    phi = []
    xi = []
    d = []
    for line in open(output_directory + filename + '_ANNE2.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        # if line.startswith('\n'):
        #   continue
        fields = line.split()
        alpha.append(fields[0])
        beta.append(fields[1])
        phi.append(fields[2])
        xi.append(fields[3])
        d.append(fields[4])
    for line in open(output_directory + filename + '_parameters.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        eta.append(fields[1])
    # Convert to Float data type
    d = [float(i) for i in d]
    alpha = [float(i) for i in alpha]
    beta = [float(i) for i in beta]
    xi = [float(i) for i in xi]
    eta = [float(i) for i in eta]
    phi = [float(i) for i in phi]

    for i in range(len(alpha)):

        x_value = d[i] * np.cos(eta[i]) * np.cos(beta[i]) + 15
        z_value = d[i] * np.cos(beta[i]) * np.cos(eta[i]) * np.tan(alpha[i])
        z_coordinates.append(z_value)

        y_value = d[i] * np.cos(beta[i]) * np.sin(eta[i])

        x_coordinates.append(x_value)
        y_coordinates.append(y_value + 13)

    with open(output_directory + filename + '_Cartesian_Coordinates.txt', 'w') as file:
    # Write Results to File
        for i in range(len(x_coordinates)):
            file.write(str(x_coordinates[i]) +
                       '\t' +
                       str(y_coordinates[i]) +
                       '\t' +
                       str(z_coordinates[i]) +
                       '\n')
    return x_coordinates, y_coordinates, z_coordinates
