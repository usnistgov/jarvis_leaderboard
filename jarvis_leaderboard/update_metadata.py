import glob, json
from jarvis.db.jsonutils import loadjson, dumpjson
from collections import defaultdict

na = []
for i in glob.glob("contributions/*/metadata.json"):
    d = loadjson(i)
    d["git_url"] = []
    if "git" in d["project_url"]:
        d["git_url"].append(d["project_url"])
    else:
        na.append(i)
    if d["team_name"] == "JARVIS":
        d["git_url"].append("https://github.com/usnistgov/jarvis")
    if d["model_name"] == "ALIGNN":
        d["git_url"].append("https://github.com/usnistgov/alignn")
    print(d)
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

print(na, len(na))
