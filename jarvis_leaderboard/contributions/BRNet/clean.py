import pandas as pd
import os
df=pd.read_csv('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip')
df['prediction']=df['prediction'].apply(lambda x:x.split('[')[1].split(']')[0])
df.to_csv('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv', index=False)
cmd='zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)
