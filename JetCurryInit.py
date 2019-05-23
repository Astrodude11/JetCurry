import sys
import imp

pythonVersion = sys.version_info[0]
if pythonVersion == 3:
  tk = "tkinter"
else:
  tk = "Tkinter"

def check_modules():
  # Required modules
  modules = ['emcee',
            'multiprocessing',
            'scipy',
            'numpy',
            'astropy',
            'matplotlib',
            'math',
            'numpy',
            'pylab',
            'argparse',
            'glob',
            'PIL',
            tk]

  '''
  Try to import required modules.
  If a module can't be imported then print module name and exit
  '''
  missing_modules = []

  for module in modules:
    try:
      imp.find_module(module)
    except ImportError:
      missing_modules.append(module)

  return missing_modules
