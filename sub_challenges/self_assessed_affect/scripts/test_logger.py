import sys
sys.path.append("/Users/bhavyakarki/Desktop/design791/last/791-Project/logs")
import log_config
from sklearn.metrics import classification_report

log_config.set_exp_name("reportRaw")

log_config.logger("Testing starts.......", 'INFO')
'''
name="A"
log_config.logger("hi there %s" % name, 'ERROR')

log_config.logger("3 + 4 = %d" %7)

score = 25
log_config.logger("Total score for %s is %d" % (name, score), 'INFO')
'''
#list_a = ["a","b","c","d","e"]
#str1 = '\n'.join(list_a)
#log_config.logger(list_a)


#log_config.logger("Today is a sunny day at " + str(float(0.038)) +  " in CMU ")

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
#log_config.logger(classification_report(y_true, y_pred, target_names=target_names),'REPORT_CR')


dev_data = {'a':(1,2), 'b':(2,3), 'c':(3,4), 'd':(4,5)}
ids = [1,2,3,4,5,6,7,8]
y_test_pred = []
for k, i in dev_data.items():
    pred = int(i[0])
    log_config.logger(str(k) + ' ' + str(pred) + ':' + str(ids[pred]) +  ' ' + str(pred) + ':' + str(ids[pred])  + '\n', 'REPORT_RAW')

