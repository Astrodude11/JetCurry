'''
Copyright 2017, Andrew Colson, Katie Kosak, Eric Perlman
JETCURRY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
__author__ = "Katie Kosak"
__email__ = "katie.kosak@gmail.com"
__credits__ = ["KunYang Li", "Dr. Eric Perlman", "Dr.Sayali Avachat", "Andrew Colson"]
__status__ = "production"

import os
import JetCurryInit

# Prints any missing modules and exits. Otherwise, continue.
requiredModules = JetCurryInit.check_modules()
if requiredModules:
    print('Module(s) ' + ', '.join(requiredModules) + ' not installed')
    os.sys.exit()

from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import JetCurry as jet
import argparse
import glob
import JetCurryGui
from JetCurryLogger import write_log
import itertools


# Create command line argument parser
parser = argparse.ArgumentParser(description="Jet Curry")
parser.add_argument('input', help='file or folder name')
parser.add_argument('-out_dir', help='output directory path')
parser.add_argument('-debug', help='console logger', action='store_true')
args = parser.parse_args()

'''
Determine whether input is a single file or directory
Create list of FITS files for processing
'''
files = []
if os.path.isfile(args.input):
    try:
        fits.PrimaryHDU.readfrom(args.input)
        files.append(args.input)
    except BaseException:
        print('%s is not a valid FITS file!' % args.input)
        os.sys.exit()
else:
    if os.path.exists(args.input):
        for file in glob.glob(args.input + '/*.fits'):
            try:
                fits.PrimaryHDU.readfrom(file)
                files.append(file)
            except BaseException:
                print('%s is not a valid FITS file!' % file)
    else:
        print('Input directory does not exist!')

# Create output directory if it doesn't exitst
if args.out_dir is not None:
    if not os.path.exists(args.out_dir):
        try:
            os.makedirs(args.out_dir)
        except BaseException:
            print('%s does not exist and cannot be created' % args.out_dir)
else:
    args.out_dir = os.getcwd()

if args.out_dir[-1] == '/':
    output_directory_default = args.out_dir
else:
    output_directory_default = args.out_dir + '/'

np.seterr(all='ignore')
S = []
ETA = []
# Line of Sight (radians)
THETA = 0.261799388

for file in files:
    filename = os.path.splitext(file)[0]
    filename = os.path.basename(filename)
    output_directory = output_directory_default + filename + '/'

    # create output directory
    # if directory already exists, append name with _N
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        for i in itertools.count(1):
            output_directory = output_directory_default + filename + '_' + str(i) + '/'
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                break

    # create log file
    log_filename = output_directory + filename + '.log'

    curry = JetCurryGui.JetCurryGui(file)
    upstream_bounds = np.array(
        [curry.x_start_variable.get(), curry.y_start_variable.get()])
    downstream_bounds = np.array(
        [curry.x_end_variable.get(), curry.y_end_variable.get()])

    if (not np.any(upstream_bounds) or not np.any(downstream_bounds)):
        print('Please select upstream and downstream bounds')
        os.sys.exit()

    write_log(log_filename, 'info', 'Using filename: ' + filename + '.fits', args.debug)
    write_log(log_filename, 'info', 'Output directory set to ' + output_directory, args.debug)
    write_log(log_filename, 'info', 'Upstream bound is: ' + str(upstream_bounds), args.debug)
    write_log(log_filename, 'info', 'Downstream bound is: ' + str(downstream_bounds), args.debug)
    
    pixel_min = np.nanmin(curry.fits_data)
    pixel_max = np.nanmax(curry.fits_data)

    # Square Root Scaling for fits image
    try:
        data = jet.imagesqrt(curry.fits_data, pixel_min, pixel_max)
    except Exception as e:
        write_log(log_filename, 'critical', 'Failed to create square root image of data:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'Created square root image of data:', args.debug)
        write_log(log_filename, 'info', str(data) + '\n', args.debug)

    number_of_points = downstream_bounds[0] - upstream_bounds[0]
    if number_of_points > 0:
        write_log(log_filename, 'info', 'Number of sample points: ' +  str(number_of_points), args.debug)
    else:
        write_log(log_filename, 'critical', 'The number of sample points must be positive: ' + str(number_of_points), args.debug)
        os.sys.exit()

    plt.imshow(data)

    # Go column by column to calculate the max flux for each column
    try:
        x, y, x_smooth, y_smooth, intensity_max = jet.Find_MaxFlux(
            data, upstream_bounds, downstream_bounds, number_of_points)
    except Exception as e:
        write_log(log_filename, 'critical', 'Failed to calculate the max intensity:', args.debug)
        write_log(log_filename, 'critical', e , args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'Successfully calculated the max intensities', args.debug)
        write_log(log_filename, 'info', 'Max intensity at point x:', args.debug)
        write_log(log_filename, 'info', str(x) + '\n', args.debug)
        write_log(log_filename, 'info', 'Max intensity at point y:', args.debug)
        write_log(log_filename, 'info', str(y) + '\n', args.debug)
        write_log(log_filename, 'info', 'Smoothed sample points calculated over the start/stop interval:', args.debug)
        write_log(log_filename, 'info', str(x_smooth) + '\n', args.debug)
        write_log(log_filename, 'info', 'Interpolated curve using spline fit:', args.debug)
        write_log(log_filename, 'info', str(y_smooth) + '\n', args.debug)
        write_log(log_filename, 'info', 'Max intensity at each column:', args.debug)
        write_log(log_filename, 'info', str(intensity_max) + '\n', args.debug)

    # Plot the Max Flux over the Image
    plt.contour(data, 10, cmap='gray')
    plt.scatter(x_smooth, y_smooth, c='b')
    plt.scatter(x, y, c='b')
    plt.title('Outline of Jet Stream')
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.show()
    plt.savefig(output_directory + filename + '_contour.png')
    plt.clf()

    # Calculate the s and eta values
    # s,eta, x_smooth,y_smooth values will
    # be stored in parameters.txt file
    try:
        S, ETA = jet.Calculate_s_and_eta(
            x_smooth, y_smooth, upstream_bounds, output_directory, filename)
    except Exception as e:
        write_log(log_filename, 'critical', 'Failed to calculate S and ETA:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
    else:
        write_log(log_filename, 'info', 'Calculated S:', args.debug)
        write_log(log_filename, 'info', str(S) + '\n', args.debug)
        write_log(log_filename, 'info', 'Calculated ETA:', args.debug)
        write_log(log_filename, 'info', str(ETA) + '\n', args.debug)

    # Run the First MCMC Trial in Parallel
    try:
        jet.MCMC1_Parallel(S, ETA, THETA, output_directory, filename)
    except Exception as e:
        write_log(log_filename, 'critical', 'MCMC1_Parallel failed:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'MCM1_Parallel passed', args.debug)

    try:
        jet.MCMC2_Parallel(S, ETA, THETA, output_directory, filename)
    except Exception as e:
        write_log(log_filename, 'critical', 'MCMC2_Parallel failed:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'MCMC2_Parallel passed', args.debug)

    # Run Simulated Annealing to guarantee Real Solution
    try:
        jet.Annealing1_Parallel(S, ETA, THETA, output_directory, filename)
    except Exception as e:
        write_log(log_filename, 'critical', 'Annealing1_Parallel failed:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'Annealing1_Parallel passed', args.debug)

    try:
        jet.Annealing2_Parallel(S, ETA, THETA, output_directory, filename)
    except:
        write_log(log_filename, 'critical', 'Annealing2_Parallel failed:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'Annealing2_Parallel passed', args.debug)

    try:
        x_coordinates, y_coordinates, z_coordinates = jet.Convert_Results_Cartesian(
            S, ETA, THETA, output_directory, filename)
    except Exception as e:
        write_log(log_filename, 'critical', 'Failed to convert cartesian coordinates:', args.debug)
        write_log(log_filename, 'critical', e, args.debug)
        os.sys.exit()
    else:
        write_log(log_filename, 'info', 'x coordinates:', args.debug)
        write_log(log_filename, 'info', str(x_coordinates) + '\n', args.debug)
        write_log(log_filename, 'info', 'y coordinates:', args.debug)
        write_log(log_filename, 'info', str(y_coordinates) + '\n', args.debug)
        write_log(log_filename, 'info', 'z coordinates:', args.debug)
        write_log(log_filename, 'info', str(z_coordinates) + '\n', args.debug)

    # Plot the Results on Image
    plt.scatter(x_coordinates, y_coordinates, c='y')
    plt.scatter(x_smooth, y_smooth, c='r')
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.show()
    plt.savefig(output_directory + filename + '_sim.png')
    plt.clf()

    write_log(log_filename, 'info', 'Jet Curry successful for ' + filename + '.fits' + '\n', args.debug)
