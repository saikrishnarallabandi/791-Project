import sys
sys.path.append("/Users/bhavyakarki/Desktop/design791/last/791-Project/logs")
import log_config

log_config.set_exp_name("listTest")

log_config.logger("Testing starts.......", 'INFO')
'''
name="A"
log_config.logger("hi there %s" % name, 'ERROR')

log_config.logger("3 + 4 = %d" %7)

score = 25
log_config.logger("Total score for %s is %d" % (name, score), 'INFO')
'''
list_a = ["a","b","c","d","e"]
#str1 = '\n'.join(list_a)
log_config.logger(list_a)