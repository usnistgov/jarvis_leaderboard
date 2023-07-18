import matplotlib.pyplot as plt
from jarvis_leaderboard.rebuild import get_metric_value, get_results
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
from jarvis_leaderboard.rebuild import get_metric_value,get_results
import glob
for ii in glob.glob("*.csv.zip"):
 print()
 print(ii)
 names,vals=get_results(bench_name=ii)
 for i,j in zip(names,vals):
     print(i,j)
     

main_dir = "/wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/alignnff_fmult_mlearn_only"

els = ["Si", "Ni", "Cu", "Mo", "Li", "Ge"]

for i in els:
    fname='AI-MLFF-forces-mlearn_'+i+'-test-multimae.csv.zip'
    forces=os.path.join(main_dir,fname)
    res = get_metric_value(forces)
    pred = np.concatenate(
        [
            np.array(i.split(";"), dtype="float")
            for i in res["df"]["prediction"].values
        ]
    )
    actual = np.concatenate(
        [
            np.array(i.split(";"), dtype="float")
            for i in res["df"]["actual"].values
        ]
    )
    print("MAE F", mean_absolute_error(actual, pred))
    plt.plot(actual, pred, ".")
    figname="ff"+i+"forces.png"
    plt.savefig(figname)
    plt.close()


    fname='AI-MLFF-energy-mlearn_'+i+'-test-mae.csv.zip'
    energy=os.path.join(main_dir,fname)
    res = get_metric_value(energy)
    print("res", res["df"])
    actual = res["df"]["actual"].values
    pred = res["df"]["prediction"].values
    print(actual, actual.shape)
    print(pred, pred.shape)
    print("MAE E", mean_absolute_error(actual, pred))
    plt.plot(actual, pred, ".")
    figname="ff"+i+"energy.png"
    plt.savefig(figname)
    #plt.savefig("ffSienergy.png")
    plt.close()


    fname='AI-MLFF-stresses-mlearn_'+i+'-test-multimae.csv.zip'
    stresses=os.path.join(main_dir,fname)
    res = get_metric_value(stresses)
    print(res["df"])
    pred = np.concatenate(
        [
            np.array(i.split(";"), dtype="float")
            for i in res["df"]["prediction"].values
        ]
    )
    actual = np.concatenate(
        [
            np.array(i.split(";"), dtype="float")
            for i in res["df"]["actual"].values
        ]
    )
    #pred = pred * (-1600 * 3)
    print("MAE S", mean_absolute_error(actual, pred))
    plt.plot(actual, pred, ".")
    figname="ff"+i+"stress.png"
    plt.savefig(figname)
    #plt.savefig("ffSistress.png")
    plt.close()



