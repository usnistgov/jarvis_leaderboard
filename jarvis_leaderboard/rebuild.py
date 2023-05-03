#! /usr/bin/env python
import os
from jarvis.db.jsonutils import loadjson
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd
import glob
import zipfile
import json
from collections import defaultdict
import numpy as np

# from mkdocs import utils

# base_url = utils.get_relative_url('.','.')
# print ('base_url',base_url)
root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
clean = True
errors = []

scaling = {
    "qm9_std_jctc": {
        "alpha": 8.172947,
        "mu": 1.503449,
        "HOMO": 0.597728,
        "LUMO": 1.274800,
        "gap": 1.284114,
        "R2": 280.472586,
        "ZPVE": 0.901645,
        "U0": 10.322918,
        "U": 10.414332,
        "H": 10.488418,
        "G": 9.497589,
        "Cv": 4.067492,
    }
}


def make_summary_table():
    methods = ["AI", "ES", "FF", "QC", "EXP"]
    tasks = [
        "SinglePropertyPrediction",
        "SinglePropertyClass",
        "MLFF",
        "TextClass",
        "ImageClass",
        "Spectra",
        "EigenSolver",
    ]
    line = '<table style="width:100%" id="j_table">'
    line += "<thead><td>Methods</td>"
    for i in tasks:
        line += "<td>" + i + "</td>"
    # line+='<td>Total</td>'

    def get_num_benches(method="AI", task="SinglePropertyPrediction"):
        num = 0

        md_file = "docs/" + method + "/" + task + "/index.md"
        if os.path.exists(md_file):
            f = open(md_file, "r")
            lines = f.read().splitlines()
            f.close()
            for i in lines:
                if "Number of contributions:" in i:
                    num = int(i.split("Number of contributions:")[1])
        return str(num)

    for i in methods:
        line += "<tr>" + "<td>" + i + "</td>"
        for j in tasks:
            num = get_num_benches(method=i, task=j)
            if num == "0":
                line += "<td>" + "-" + "</td>"
            else:
                line += (
                    '<td><a href="./'
                    + i
                    + "/"
                    + j
                    + '" target="_blank">'
                    + num
                    + "</a></td>"
                )

        line += "<tr>"

    line += "</table>"

    md_path = "docs/index.md"
    with open(md_path, "r") as file:
        filedata = file.read().splitlines()
    content = []
    for j in filedata:
        if "<!--summary_table-->" in j:
            content.append("<!--summary_table-->")
        # elif "<!--number_of_methods-->" in j:
        #    content.append("<!--number_of_methods-->")
        else:
            content.append(j)
    with open(md_path, "w") as file:
        file.write("\n".join(content))

    with open(md_path, "r") as file:
        filedata = file.read().splitlines()
    content = []

    for j in filedata:
        if "<!--summary_table-->" in j:
            temp = j + line
            content.append(temp)
        else:
            content.append(j)
    # filedata = filedata.replace('<!--table_content-->', temp)

    with open(md_path, "w") as file:
        file.write("\n".join(content))


def get_metric_value(
    submod="",
    csv_path="",
    dataset="",
    prop="",
    data_split="",
    method="",
    metric="",
    bench_name="",
):
    results = {}
    results["method"] = method
    results["submod"] = submod
    results["dataset"] = dataset
    results["prop"] = prop
    results["data_split"] = data_split
    results["csv_path"] = csv_path
    results["metric"] = metric
    csv_data = pd.read_csv(csv_path, sep=",")
    meta_path = csv_path.split("/")
    meta_path[-1] = "metadata.json"
    meta_path = "/".join(meta_path)
    meta_data = loadjson(meta_path)
    results["model_name"] = meta_data["model_name"]
    results["team_name"] = meta_data["team_name"]
    results["date_submitted"] = meta_data["date_submitted"]
    results["project_url"] = meta_data["project_url"]
    results["num_entries"] = len(csv_data)
    results["github_url"] = (
        "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/"
        #"https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/"
        + bench_name
    )  # meta_path.split('metadata.json')[0]

    # print("meta_path", meta_data)
    # meta_data=loadjson()
    # print("csv_data", csv_path)
    # dataset with actual values
    temp = dataset + "_" + prop + ".json"
    # print ('json temp',temp)
    temp2 = temp + ".zip"
    fname = os.path.join("benchmarks", method, submod, temp2)
    #fname = os.path.join("dataset", method, submod, temp2)
    fname2 = os.path.join(root_dir, fname)

    z = zipfile.ZipFile(fname2)
    json_data = json.loads(z.read(temp))

    # json_data = loadjson(os.path.join(root_dir, fname))
    actual_data_json = json_data[data_split]
    if "val" in json_data:  # sometimes just train-test
        data_size = (
            len(json_data["train"])
            + len(json_data["val"])
            + len(json_data["test"])
        )
    else:
        data_size = len(json_data["train"]) + len(json_data["test"])
    # print ('actual_data_json',actual_data_json)
    results["dataset_size"] = data_size
    ids = []
    targets = []
    for i, j in actual_data_json.items():
        ids.append(i)
        targets.append(j)
    mem = {"id": ids, "actual": targets}
    actual_df = pd.DataFrame(mem)
    # print ('actual_df',actual_df)
    # print('csv_data',csv_data)
    # actual_df.to_csv('actual_df.csv')
    # csv_data.to_csv('csv_data.csv')
    csv_data["id"] = csv_data["id"].astype(str)
    actual_df["id"] = actual_df["id"].astype(str)
    if len(csv_data) != len(actual_df):
        print("Error", csv_path, len(csv_data), len(actual_df))
        errors.append(csv_path)

    df = pd.merge(csv_data, actual_df, on="id")
    # print('csv',csv_path)
    # print ('df',df)
    # print('csv_data',csv_data)
    # print('actual_df',actual_df)
    results["res"] = "na"
    if metric == "mae":
        res = round(mean_absolute_error(df["actual"], df["prediction"]), 3)
        results["res"] = res
        if "qm9_std_jctc" in csv_path:
            # print('scaling[dataset][prop],',scaling[dataset][prop])
            res = round(
                scaling[dataset][prop]
                * mean_absolute_error(df["actual"], df["prediction"]),
                3,
            )
            results["res"] = res
            # print(csv_path)
            # print('mae1',mean_absolute_error(csv_data['target'],csv_data['prediction']))
            # print('res',res)
            # print(csv_data)
            # print(actual_df)
            # print()
    if metric == "acc":
        # print("ACC",csv_path)
        # print(df, len(df))
        res = round(accuracy_score(df["actual"], df["prediction"]), 3)
        # print("res", res)
        results["res"] = res
    if metric == "multimae":
        # print("csv multimae", csv_path)
        # print ('df',df)
        maes = []
        for k, v in df.iterrows():
            real = np.array(v["actual"].split(";"), dtype="float")
            # real = np.array(v["target"].split(";"), dtype="float")
            pred = np.array(v["prediction"].split(";"), dtype="float")
            m = mean_absolute_error(real, pred)
            maes.append(m)
            # print('mm',m)
        results["res"] = round(np.array(maes).sum(), 3)
        # print ('df',df)
        # print('csv_data',csv_data)
        # print('actual_df',actual_df)
        # print('res',results['res'])

    return results


def rebuild_pages():
    print("Rebuilding web:")
    unique_fname = []
    os.chdir(root_dir + "/..")
    num_data = 0
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
    #for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
        # if 'Text' in i:
        print(i)
        fname = i.split("/")[-1].split(".csv.zip")[0]
        temp = fname.split("-")
        submod = temp[1]
        data_split = temp[4]
        prop = temp[2]
        dataset = temp[3]
        method = temp[0]
        metric = temp[-1]
        # print ('metric',metric)
        # print ('dataset',dataset)
        team = i.split("/")[-2]
        md_filename = "../docs/" + method + "/" + submod + "/" + prop + ".md"
        md_filename = (
            "../docs/"
            + method
            + "/"
            + submod
            + "/"
            + dataset
            + "_"
            + prop
            + ".md"
        )
        # print ('md_filename',md_filename)
        md_path = os.path.join(root_dir, md_filename)
        # print(
        #    fname,
        #    data_split,
        #    prop,
        #    dataset,
        #    method,
        #    metric,
        #    team,
        #    md_filename,
        #    md_path,
        # )
        with open(md_path, "r") as file:
            filedata = file.read().splitlines()
        content = []
        for j in filedata:
            if "<!--table_content-->" in j:
                content.append("<!--table_content-->")
            else:
                content.append(j)
        with open(md_path, "w") as file:
            file.write("\n".join(content))
    # jarvis_leaderboard/dataset/AI/dft_3d_exfoliation_energy.json
    dat = []
    md_files = []
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
    #for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
        print(i)
        fname = i.split("/")[-1].split(".csv.zip")[0]
        bench_name = i.split("/")[-2]
        temp = fname.split(
            "-"
        )  # ['SinglePropertyPrediction', 'test', 'bandgap', 'dft_3d_JVASP_1002_Si', 'ES', 'mae']
        # submod = temp[0]
        # data_split = temp[1]
        # prop = temp[2]
        # dataset = temp[3]
        # method = temp[4]
        # metric = temp[5]
        # method = temp[-2]
        # metric = temp[-1]

        submod = temp[1]
        data_split = temp[4]
        prop = temp[2]
        dataset = temp[3]
        method = temp[0]
        metric = temp[-1]

        team = i.split("/")[-2]
        md_filename = "../docs/" + method + "/" + submod + "/" + prop + ".md"
        md_filename = (
            "../docs/"
            + method
            + "/"
            + submod
            + "/"
            + dataset
            + "_"
            + prop
            + ".md"
        )
        md_path = os.path.join(root_dir, md_filename)
        md_files.append(md_path)
        notes = ""
        notes = (
            '<a href="'
            + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/"
            + i
            + '" target="_blank">CSV</a>'
        )
        json_name = dataset + "_" + prop + ".json.zip"
        json_path = method + "/" + submod + "/" + json_name
        json_url = (
            '<a href="'
            + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/"
            #+ "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/dataset/"
            + json_path
            + '" target="_blank">JSON</a>'
        )
        metadata = (
            '<a href="'
            + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/"
            + "jarvis_leaderboard/contributions/"
            #+ "jarvis_leaderboard/benchmarks/"
            + bench_name
            + "/metadata.json "
            + '" target="_blank">Info</a>'
        )
        runsh = (
            '<a href="'
            + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/"
            + "jarvis_leaderboard/contributions/"
            #+ "jarvis_leaderboard/benchmarks/"
            + bench_name
            + "/run.sh "
            + '" target="_blank">run.sh</a>'
        )
        notes = notes + ", " + json_url + ", " + runsh + ", " + metadata
        if "JVASP" in prop:
            jid = "JVASP-" + prop.split("JVASP_")[1].split("_")[0]
            #print("propjid", prop, jid)
            jid_url = (
                '<a href="'
                + "https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/"
                + jid
                + ".xml "
                + '" target="_blank">'
                + jid
                + "</a>"
            )
            notes += ", " + jid_url

        # print ('bench_name', bench_name)
        # print(
        #    fname,
        #    data_split,
        #    prop,
        #    dataset,
        #    method,
        #    metric,
        #    team,
        #    md_filename,
        #    md_path,
        # )
        with open(md_path, "r") as file:
            filedata = file.read().splitlines()
        # print("names", i)
        # print()
        res = get_metric_value(
            submod=submod,
            csv_path=i,
            dataset=dataset,
            prop=prop,
            data_split=data_split,
            method=method,
            metric=metric,
            bench_name=bench_name,
        )
        if fname not in unique_fname:
            unique_fname.append(fname)
            num_data += res["dataset_size"]
        # num_data+=res['dataset_size']
        # num_data+=res['num_entries']
        print("num_data", i, num_data, res["dataset_size"], res["num_entries"])
        # res = 5
        # if clean:

        team = (
            '<a href="'
            + res["github_url"]
            + '" target="_blank">'
            + team
            + "</a>"
            # '<a href="' + res["project_url"] + '" target="_blank">' + team + "</a>"
        )
        # team='['+team+']'+'('+res['project_url']+')'
        info = {}
        temp = (
            "<!--table_content-->"
            + "<tr>"
            + "<td>"
            + team
            + "</td>"
            + "<td>"
            + dataset
            + "</td>"
            # + "<td>"
            # + method
            # + "</td>"
            + "<td>"
            + str(res["res"])
            + "</td>"
            # + "<td>"
            # + str(res['model_name'])
            # + "</td>"
            + "<td>"
            + str(res["team_name"])
            + "</td>"
            + "<td>"
            + str(res["dataset_size"])
            + "</td>"
            + "<td>"
            + str(res["date_submitted"])
            + "</td>"
            + "<td>"
            + str(notes)
            + "</td>"
            + "</tr>"
        )
        info["team"] = team
        info["result"] = res
        dat.append(info)
        content = []
        for j in filedata:
            if "<!--table_content-->" in j:
                temp = temp + j
                content.append(temp)
            else:
                content.append(j)
        # filedata = filedata.replace('<!--table_content-->', temp)

        with open(md_path, "w") as file:
            file.write("\n".join(content))
    # print("dat", dat)
    print("mdfiles", len(set(md_files)))

    def update_individual_index_md(
        md_path="docs/ES/index.md", key="ES", extra_key="-", homepage=[]
    ):
        n_methods = 0
        for i in glob.glob("jarvis_leaderboard/contributions/*/metadata.json"):
        #for i in glob.glob("jarvis_leaderboard/benchmarks/*/metadata.json"):
            n_methods += 1
        if not homepage:
            homepage = []
            for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
            #for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):

                if key in i and extra_key in i:
                    p = i.split("/")[-1].split(".csv.zip")[0]
                    homepage.append(p)
        # print ('index pages',homepage)
        # print("dat", dat)
        # print("errors", errors, len(errors))
        selected = defaultdict()
        for name in homepage:
            # print(md_path,name)
            for i in dat:
                # name2 = (
                #    i["result"]["submod"]
                #    + "-"
                #    + i["result"]["data_split"]
                #    + "-"
                #    + i["result"]["prop"]
                #    + "-"
                #    + i["result"]["dataset"]
                #    + "-"
                #    + i["result"]["method"]
                #    + "-"
                #    + i["result"]["metric"]
                # )
                name2 = (
                    i["result"]["method"]
                    + "-"
                    + i["result"]["submod"]
                    + "-"
                    + i["result"]["prop"]
                    + "-"
                    + i["result"]["dataset"]
                    + "-"
                    + i["result"]["data_split"]
                    + "-"
                    + i["result"]["metric"]
                )
                if name == name2:
                    temp = float(i["result"]["res"])
                    # if md_path!='docs/index.md':
                    # selected[name] = i["result"]
                    i["result"]["team"] = i["team"]
                    if name not in selected:
                        selected[name] = i["result"]
                    elif (
                        temp > selected[name]["res"]
                        and i["result"]["metric"] == "acc"
                    ):
                        selected[name] = i["result"]
                    elif (
                        temp < selected[name]["res"]
                        and i["result"]["metric"] == "mae"
                    ):
                        selected[name] = i["result"]
                    elif (
                        temp < selected[name]["res"]
                        and i["result"]["metric"] == "multimae"
                    ):
                        selected[name] = i["result"]
        temp = (
            '<!--table_content--><table style="width:100%" id="j_table">'
            + "<thead><tr>"
            + "<th>Method</th>"
            # +'<th><a href="./method' + '" target="_blank">' + 'Method' + "</a></th>"
            + "<th>Task</th>"
            + "<th>Property</th>"
            + "<th>Model name</th>"
            + "<th>Metric</th>"
            + "<th>Score</th>"
            + "<th>Team</th>"
            + "<th>Dataset</th>"
            + "<th>Size</th>"
            + "</tr></thead>"
        )
        for i, j in selected.items():
            if len(md_path.split("/")) == 2:
                # if md_path == "docs/index.md":
                temp = (
                    temp
                    + "<tr>"
                    + "<td>"
                    + '<a href="./'
                    + j["method"]
                    + '" target="_blank">'
                    + j["method"]
                    + "</a>"
                    # + j["method"]
                    + "</td>"
                    + "<td>"
                    + '<a href="./'
                    # + j["dataset"]+'_'
                    + j["method"]
                    + "/"
                    + j["submod"]
                    + '" target="_blank">'
                    + j["submod"]
                    + "</a>"
                    # + j["submod"]
                    + "</td>"
                    + "<td>"
                    + '<a href="./'
                    + j["method"]
                    + "/"
                    + j["submod"]
                    + "/"
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + '" target="_blank">'
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + "</a>"
                    # + j["prop"]
                    + "</td>"
                    + "<td>"
                    + j["team"]
                    + "</td>"
                    + "<td>"
                    + str(j["metric"].upper())
                    + "</td>"
                    + "<td>"
                    + str(j["res"])
                    + "</td>"
                    + "<td>"
                    + str(j["team_name"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset_size"])
                    + "</td>"
                    # + "<td>"
                    # + str(j["date_submitted"])
                    # + "</td>"
                    + "</tr>"
                )
            elif len(md_path.split("/")) == 3:
                base = "."
                temp = (
                    temp
                    + "<tr>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    + j["method"]
                    + '" target="_blank">'
                    + j["method"]
                    + "</a>"
                    # + j["method"]
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    # + j["method"]
                    # + "/"
                    + j["submod"]
                    + '" target="_blank">'
                    + j["submod"]
                    + "</a>"
                    # + j["submod"]
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    # + j["method"]
                    # + "/"
                    + j["submod"]
                    + "/"
                    # + "/"
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + '" target="_blank">'
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + "</a>"
                    # + j["prop"]
                    + "</td>"
                    + "<td>"
                    + j["team"]
                    + "</td>"
                    + "<td>"
                    + str(j["metric"].upper())
                    + "</td>"
                    + "<td>"
                    + str(j["res"])
                    + "</td>"
                    + "<td>"
                    + str(j["team_name"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset_size"])
                    + "</td>"
                    # + "<td>"
                    # + str(j["date_submitted"])
                    # + "</td>"
                    + "</tr>"
                )
            elif len(md_path.split("/")) == 4:
                base = "."
                temp = (
                    temp
                    + "<tr>"
                    + "<td>"
                    # + '<a href= ".'+base+'/'
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    + j["method"]
                    + '" target="_blank">'
                    + j["method"]
                    + "</a>"
                    # + j["method"]
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    + j["method"]
                    + "/"
                    + j["submod"]
                    + '" target="_blank">'
                    + j["submod"]
                    + "</a>"
                    # + j["submod"]
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    # + j["method"]
                    # + "/"
                    # + j["submod"]
                    # + "/"
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + '" target="_blank">'
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + "</a>"
                    # + j["prop"]
                    + "</td>"
                    + "<td>"
                    + j["team"]
                    + "</td>"
                    + "<td>"
                    + str(j["metric"].upper())
                    + "</td>"
                    + "<td>"
                    + str(j["res"])
                    + "</td>"
                    + "<td>"
                    + str(j["team_name"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset"])
                    + "</td>"
                    + "<td>"
                    + str(j["dataset_size"])
                    + "</td>"
                    # + "<td>"
                    # + str(j["date_submitted"])
                    # + "</td>"
                    + "</tr>"
                )

        # md_path = "docs/index.md"
        # print (md_path,temp)

        with open(md_path, "r") as file:
            filedata = file.read().splitlines()
        content = []
        for j in filedata:
            if "<!--table_content-->" in j:
                content.append("<!--table_content-->")
            elif "<!--number_of_contributions-->" in j:
                content.append("<!--number_of_contributions-->")
            elif "<!--number_of_benchmarks-->" in j:
                content.append("<!--number_of_benchmarks-->")
            elif "<!--number_of_methods-->" in j:
                content.append("<!--number_of_methods-->")
            elif "<!--number_of_datapoints-->" in j:
                content.append("<!--number_of_datapoints-->")
            else:
                content.append(j)
        with open(md_path, "w") as file:
            file.write("\n".join(content))

        with open(md_path, "r") as file:
            filedata = file.read().splitlines()
        content = []
        n_benchs = len(homepage)
        if md_path == "docs/index.md":
            n_benchs = len(dat)

        for j in filedata:
            if "<!--table_content-->" in j:
                temp = temp + j + "</table>"
                content.append(temp)
            elif "<!--number_of_benchmarks-->" in j:
                temp2 = (
                    "<!--number_of_benchmarks--> - Number of benchmarks: "
                    + str(len(set(md_files)))
                    # + str(len(dat))
                    # + "\n"
                )
                content.append(temp2)
            elif "<!--number_of_methods-->" in j:
                temp2 = (
                    # "<!--number_of_methods--> - Number of methods: "
                    "<!--number_of_methods--> - Number of methods: "
                    + "["
                    + str(n_methods)
                    + "]"
                    + "(https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions)"
                    # + str(n_methods)
                    # + str(len(dat))
                    # + "\n"
                )
                content.append(temp2)
            elif "<!--number_of_contributions-->" in j:
                temp2 = (
                    "<!--number_of_contributions--> - Number of contributions: "
                    + str(n_benchs)
                    # + str(len(dat))
                    # + "\n"
                )
                content.append(temp2)
            elif "<!--number_of_datapoints-->" in j:
                temp2 = (
                    "<!--number_of_datapoints--> - Number of datapoints: "
                    + str(num_data)
                    # + str(num_data)
                    # + str(len(dat))
                    # + "\n"
                )
                content.append(temp2)
            else:
                content.append(j)
        # filedata = filedata.replace('<!--table_content-->', temp)

        with open(md_path, "w") as file:
            file.write("\n".join(content))

    homepage = [
        "AI-SinglePropertyPrediction-formula_energy-ssub-test-mae",
        "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae",
        "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae",
        "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae",
        "AI-SinglePropertyPrediction-optb88vdw_total_energy-dft_3d-test-mae",
        "AI-SinglePropertyPrediction-bulk_modulus_kv-dft_3d-test-mae",
        "AI-SinglePropertyClass-optb88vdw_bandgap-dft_3d-test-acc",
        "AI-SinglePropertyPrediction-LUMO-qm9_std_jctc-test-mae",
        "AI-SinglePropertyPrediction-max_co2_adsp-hmof-test-mae",
        "AI-MLFF-energy-alignn_ff_db-test-mae",
        "AI-ImageClass-bravais_class-stem_2d_image-test-acc",
        "AI-TextClass-categories-arXiv-test-acc",
        "FF-SinglePropertyPrediction-bulk_modulus_JVASP_816_Al-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-bulk_modulus_JVASP_816_Al-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-bulk_modulus-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-bulk_modulus_JVASP_1002_Si-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-bandgap-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-epsx-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-Tc_supercon_JVASP_1151_MgB2-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-Tc_supercon-dft_3d-test-mae",
        "ES-SinglePropertyPrediction-slme-dft_3d-test-mae",
        "ES-Spectra-dielectric_function-dft_3d-test-multimae",
        "QC-EigenSolver-electron_bands_JVASP_816_Al_WTBH-dft_3d-test-multimae",
        "EXP-Spectra-XRD_JVASP_19821_MgB2-dft_3d-test-multimae",
        "EXP-Spectra-co2_RM_8852-nist_isodb-test-multimae",
    ]
    x = []
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
    #for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
        x.append(i.split(".csv.zip")[0])
        # x.append(i.split('/')[-1].split('.csv.zip')[0])
    print('Files', len(x))
    #print(x, len(x))
    update_individual_index_md(md_path="docs/index.md", homepage=homepage)
    # update_individual_index_md(md_path="docs/index.md",homepage=sorted(x))
    update_individual_index_md(md_path="docs/ES/index.md", key="ES")
    update_individual_index_md(md_path="docs/FF/index.md", key="FF")
    update_individual_index_md(
        md_path="docs/ES/SinglePropertyPrediction/index.md",
        key="ES",
        extra_key="SinglePropertyPrediction",
    )
    update_individual_index_md(
        md_path="docs/FF/SinglePropertyPrediction/index.md",
        key="FF",
        extra_key="SinglePropertyPrediction",
    )
    update_individual_index_md(
        md_path="docs/ES/Spectra/index.md", key="ES", extra_key="Spectra"
    )
    update_individual_index_md(md_path="docs/AI/index.md", key="AI")
    update_individual_index_md(
        md_path="docs/AI/SinglePropertyPrediction/index.md",
        key="AI",
        extra_key="SinglePropertyPrediction",
    )
    update_individual_index_md(
        md_path="docs/AI/SinglePropertyClass/index.md",
        key="AI",
        extra_key="SinglePropertyClass",
    )
    update_individual_index_md(
        md_path="docs/AI/MLFF/index.md", key="AI", extra_key="MLFF"
    )
    update_individual_index_md(
        md_path="docs/AI/ImageClass/index.md", key="AI", extra_key="ImageClass"
    )
    update_individual_index_md(
        md_path="docs/AI/TextClass/index.md", key="AI", extra_key="TextClass"
    )
    update_individual_index_md(md_path="docs/QC/index.md", key="QC")
    update_individual_index_md(
        md_path="docs/QC/EigenSolver/index.md",
        key="QC",
        extra_key="EigenSolver",
    )
    update_individual_index_md(md_path="docs/EXP/index.md", key="EXP")
    update_individual_index_md(
        md_path="docs/EXP/Spectra/index.md", key="EXP", extra_key="Spectra"
    )
    update_individual_index_md(
        md_path="docs/AI/Spectra/index.md", key="AI", extra_key="Spectra"
    )
    make_summary_table()
    print("unique csv names", len(unique_fname))
    print("errors", errors)
    os.chdir(current_dir)
    return errors
    # print("dat", dat)


if __name__ == "__main__":
    rebuild_pages()
