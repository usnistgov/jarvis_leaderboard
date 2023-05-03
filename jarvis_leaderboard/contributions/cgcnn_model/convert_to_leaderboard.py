#/wrk/knc6/version_tests/tests/ALL_DATASETS/CGCNN
import glob, csv,os
import pandas as pd

for i in glob.glob("*/cgcnn/*.csv"):
    print(i)
    mem = []
    with open(i, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            info = {}

            info["id"] = row[0]
            info["target"] = row[1]
            info["prediction"] = row[2]
            mem.append(info)
        df = pd.DataFrame(mem)
        fname = "SinglePropertyPrediction-test-" + i.split("/")[0] + "-dft_3d-AI-mae.csv"
        print(fname)
        fname=fname.replace('n-S','n_S').replace('n-po','n_po')
        df.to_csv(fname,index=False)
        fname1=fname+'.zip'
        cmd='zip '+fname1+' '+fname
        os.system(cmd)
        cmd='rm '+fname
        os.system(cmd)
        print(df)
