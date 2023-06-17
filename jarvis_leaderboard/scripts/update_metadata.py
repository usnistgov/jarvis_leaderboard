import glob, json
from jarvis.db.jsonutils import loadjson, dumpjson

for i in glob.glob("contributions/*/metadata.json"):
    d = loadjson(i)
    f = open(i, "w")
    f.write(json.dumps(d, indent=4))
    f.close()
    """
   if 'harware_used' in d:
    print(i)
    d['hardware_used']=d['harware_used']
    d.pop('harware_used')
    print(d)
    dumpjson(filename=i,data=d)
   """
