import logging
import os
from datetime import datetime



def set_exp_name(expname):

    # set absolute path
    dirname = "/Users/bhavyakarki/Desktop/design791/last/791-Project/logs/"
    #print __file__


    LOG_FILE_ERROR = os.path.join(dirname, 'errors/')
    LOG_FILE_REPORT = os.path.join(dirname, 'reports/')
    LOG_FILE_INFO = os.path.join(dirname, 'info/')

    setup_logger('log_error', LOG_FILE_ERROR + "error_" + expname + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )
    
    setup_logger('log_info',LOG_FILE_INFO + "info_" + expname + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )
    
    # TBD
    #setup_logger('log_report', LOG_FILE_REPORT)


def setup_logger(logger_name, log_file, level=logging.INFO):

    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


def logger(msg, level='INFO'):
 
    if level == 'INFO'    :
        log = logging.getLogger('log_info')
        log.info(msg)
    if level == 'ERROR'    :
        log = logging.getLogger('log_error')
        log.error(msg)
	 
    if level == 'WARNING' : 
        log = logging.getLogger('log_error')
        log.warning(msg)

    
