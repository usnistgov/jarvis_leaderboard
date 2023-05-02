from chemnlp.classification.scikit_class import sk_class
import time
import os
t1=time.time()
sk_class(   csv_path='/wrk/knc6/AtomNLP/Summarize/cond_mat.csv',key='categories',value='title_abstract')
t2=time.time()
print ('Time',t2-t1)
cmd = 'mv pred_test.csv '+'TextClass-test-categories-arXiv-AI-acc.csv'
os.system(cmd)

t1=time.time()
sk_class(   csv_path='/wrk/knc6/AtomNLP/Summarize/pubchem.csv',key='label_name',value='title_abstract')
t2=time.time()
print ('Time',t2-t1)
cmd = 'mv pred_test.csv '+'TextClass-test-categories-pubchem-AI-acc.csv'
os.system(cmd)
