import sys
sys.path.append("/Users/bhavyakarki/Desktop/design791/last/791-Project/logs")
import log_config

log_config.set_exp_name("testingLogger")

log_config.logger("Testing starts.......", 'INFO')

name="A"
log_config.logger("hi there %s" % name, 'ERROR')

score = 25
log_config.logger("Total score for %s is %d" % (name, score), 'INFO')