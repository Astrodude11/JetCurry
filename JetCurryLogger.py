import sys
import logging

pythonVersion = sys.version_info[0]

def write_log(logfile, logLevel, message, logMode):
	FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
	if pythonVersion == 3:
		if (logMode is True or logMode == 'True'):
			logging.basicConfig(
    			level=logging.INFO,
    			format=FORMAT,
    			handlers=[
        			logging.FileHandler(logfile),
        			logging.StreamHandler(sys.stdout)
    			])
		else:
			logging.basicConfig(
                        level=logging.INFO,
                        format=FORMAT,
                        handlers=[
							logging.FileHandler(logfile)
			])

		logger = logging.getLogger()
		if logLevel == 'info':
			logger.info(message)
		elif logLevel =='critical':
			logger.critical(message)
		else:
			logger.error(message)
	else:
		logger = logging.getLogger()
		logFormatter = logging.Formatter(FORMAT)
		logger.setLevel(logging.INFO)

		fileHandler = logging.FileHandler(logfile)
		fileHandler.setFormatter(logFormatter)
		logger.addHandler(fileHandler)

		if (logMode is True or logMode == 'True'):
			consoleHandler = logging.StreamHandler(sys.stdout)
			consoleHandler.setFormatter(logFormatter)
			logger.addHandler(consoleHandler)

		if logLevel == 'info':
			logger.info(message)
		elif logLevel =='critical':
			logger.critical(message)
		else:
			logger.error(message)
