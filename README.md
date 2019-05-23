# Jet Curry

Models the 3D geometry of AGN jets.

JetCurry, Copyright 2017. Andrew Colson. Katie Kosak. Eric Perlman. Copyright Details: GPL.txt

Last Edited: May 2019. Questions: Contact Katie Kosak,  [katie.kosak@gmail.com](mailto:katie.kosak@gmail.com)


## Usage

python JetCurryMain.py input [-out_dir] 

**Required arguments**

**input**: a single FITS file or directory (full or relative path) 

**Optional arguments** 

**-out\_dir**: full or relative path to save output files.  Output directory is created if it doesn't exist. If it already exists then the full path is appended with "\_N" where N is 1 through infinity. Default output is current working directory if -out\_dir is not specified.

**-debug**: enables logging to the console and logfile. Default behavior is to only log to the logfile saved as "inputfilename.log". The logfile is saved to the default or specified output directory. 

**Example**
> python JetCurryMain.py ./KnotD\_Radio.fits # processes single FITS file and saves data products to the current working directory

> python JetCurryMain.py ./data -out_dir /foo/bar/ # processes entire ./data directory with data products saved to specified output directory

> python JetCurryMain.py ./KnotD\_Radio.fits -debug # processes single FITS file and logs information to the console and to KnotD\_Radio.log

## Notes

A GUI will display the FITS image. Click on the image with your left mouse button to choose the upstream bounds starting point and right click to choose the downstream bounds. You may continuously click on the image until you are satisifed with the regions of interest. These values can also be entered by typing. Once satisifed, click the Run button to process the data.

Data products are organized by the FITS filename. For example, if the output directory is /foo/bar and the filename is KnotD_Radio.fits, then data products will be saved to /foo/bar/KnotD_Radio. 

## TODO

scipy.interpolate.spline is depreciated in newer versions of SciPy.
