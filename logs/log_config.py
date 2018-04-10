import logging
import os
from datetime import datetime



def set_exp_name(expname):

    # set absolute path
    dirname = "/Users/bhavyakarki/Desktop/design791/last/791-Project/logs/"
    #print __file__


    LOG_FILE_ERROR = os.path.join(dirname, 'errors/')
    LOG_FILE_REPORT_CR = os.path.join(dirname, 'reports/classification_reports/')
    LOG_FILE_REPORT_RAW = os.path.join(dirname, 'reports/raw/')
    LOG_FILE_REPORT_FINAL = os.path.join(dirname, 'reports/final_results/')
    LOG_FILE_INFO = os.path.join(dirname, 'info/')

    setup_logger('log_error', LOG_FILE_ERROR + "error_" + expname + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )
    
    setup_logger('log_info',LOG_FILE_INFO + "info_" + expname + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )
    
    # TBD
    setup_logger('log_report_cr', LOG_FILE_REPORT_CR + "report_cr_" + expname + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )

    setup_logger('log_report_raw', LOG_FILE_REPORT_RAW + "report_raw_" + expname + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )

    setup_logger('log_report_fmt', LOG_FILE_REPORT_FINAL + "report_final_" + expname + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) )


def setup_logger(logger_name, log_file, level=logging.INFO):

    log_setup = logging.getLogger(logger_name)
    if 'log_report' in logger_name:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)




def logger(msg, level='INFO'):
 
    if level == 'INFO'  or level == 'DEBUG'  :
        log = logging.getLogger('log_info')
        if isinstance(msg, (list,)): 
            log.info("Enumerating LIST")
            str1 = '\n'.join(msg)
            log.info(str1)
        else:
            log.info(msg)
    
    if level == 'ERROR'    :
        log = logging.getLogger('log_error')
        log.error(msg)
	 
    if level == 'WARNING' : 
        log = logging.getLogger('log_error')
        log.warning(msg)

    if level == 'REPORT_CR':
        log = logging.getLogger('log_report_cr')
        if isinstance(msg, (list,)): 
            log.info("Enumerating LIST")
            str1 = '\n'.join(msg)
            log.info(str1)
        else:
            log.info(msg)


    if level == 'REPORT_RAW' :
        log = logging.getLogger('log_report_raw')
        if isinstance(msg, (list,)): 
            log.info("Enumerating LIST")
            str1 = '\n'.join(msg)
            log.info(str1)
        else:
            log.info(msg)

    if level == 'REPORT_FINAL' :
        log = logging.getLogger('log_report_fmt')
        if isinstance(msg, (list,)): 
            log.info("Enumerating LIST")
            str1 = '\n'.join(msg)
            log.info(str1)
        else:
            log.info(msg)



    
