import sys
sys.path.append("/Users/bhavyakarki/Desktop/design791/last/791-Project/logs")
import log_config

log_config.set_exp_name("testingLogger")

log_config.logger("Testing starts.......", 'INFO')

name="person"
log_config.logger("hi there %s" % name, 'ERROR')