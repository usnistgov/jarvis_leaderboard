#https://adsorption.nist.gov/isodb/index.php?DOI=10.1007/s10450-018-9958-x#biblio
#https://www.nist.gov/mml/fact/reference-isotherm-co2zsm-5
from jarvis.db.jsonutils import loadjson,dumpjson
#d=loadjson("10.1007s10450-018-9958-x.Lab01.json")
d=loadjson("10.1007s10450-018-9958-x.LogisticModel.json")
pressures=[i['pressure'] for i in d["isotherm_data"]]
co2_data=[i['species_data'][0]['adsorption'] for i in d["isotherm_data"]]
co2_str=";".join(map(str,co2_data))
info={}
info["url"]="https://adsorption.nist.gov/isodb/api/isotherm/10.1007s10450-018-9958-x.Lab01.json"
info["train"]={}
info["test"]={"RM-8852":co2_str}
info["DOI"]="10.1007/s10450-018-9958-x"
dumpjson(data=info,filename="nist_isodb_co2_RM_8852.json")
for i,j in zip(pressures,co2_data):
   print (i,j)

