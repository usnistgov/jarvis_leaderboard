from chemnlp.classification.scikit_class import sk_class
import time
import os
from sklearn.svm import LinearSVC


t1=time.time()
sk_class(   csv_path='/wrk/knc6/AtomNLP/Summarize/cond_mat.csv',key='categories',model=LinearSVC(), value='title')
t2=time.time()
print ('Time',t2-t1)
cmd = 'mv pred_test.csv '+'TextClass-test-categories-arXiv-AI-acc.csv'
os.system(cmd)

t1=time.time()
sk_class(   csv_path='/wrk/knc6/AtomNLP/Summarize/pubchem.csv',key='label_name',model=LinearSVC(), value='title')
t2=time.time()
print ('Time',t2-t1)
cmd = 'mv pred_test.csv '+'TextClass-test-categories-pubchem-AI-acc.csv'
os.system(cmd)
