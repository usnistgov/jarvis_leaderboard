# https://adsorption.nist.gov/isodb/index.php?DOI=10.1007/s10450-018-9958-x#biblio
# https://www.nist.gov/mml/fact/reference-isotherm-co2zsm-5
from jarvis.db.jsonutils import loadjson, dumpjson
import requests
import os
from jarvis.core.spectrum import Spectrum

press=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40, 40.5, 41, 41.5, 42, 42.5, 43, 43.5, 44, 44.5, 45]

def get_csv(lab="Lab01"):
    fname = "10.1007s10450-018-9958-x." + lab + ".json"
    url = "https://adsorption.nist.gov/isodb/api/isotherm/" + fname
    r = requests.get(url)

    open(fname, "wb").write(r.content)

    d = loadjson(fname)
    # d=loadjson("10.1007s10450-018-9958-x.LogisticModel.json")
    pressures = [i["pressure"] for i in d["isotherm_data"]]
    print(pressures)
    csv_name = "EXP-Spectra-co2_RM_8852-nist_isodb-test-multimae.csv"
    f = open(csv_name, "w")
    co2_data = [i["species_data"][0]["adsorption"] for i in d["isotherm_data"]]
    print (len(co2_data))

    s = Spectrum(x=pressures,y=co2_data)
    co2_data = s.get_interpolated_values(new_dist=press)
    co2_str = ";".join(map(str, co2_data))
    line = "id,prediction\n"
    f.write(line)
    line = "RM-8852" + "," + co2_str + "\n"
    f.write(line)
    f.close()
    cmd = "zip " + csv_name + ".zip " + csv_name
    os.system(cmd)
    cmd = "rm " + csv_name
    os.system(cmd)
    cmd = "rm " + fname
    os.system(cmd)


lab="LogisticModel"
lab="Lab05"
#get_csv(lab)
get_csv(lab)
