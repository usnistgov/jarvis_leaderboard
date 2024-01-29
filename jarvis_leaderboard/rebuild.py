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
import matplotlib.pyplot as plt
import requests
from jarvis.io.vasp.inputs import Poscar
import plotly.graph_objects as go

# from mkdocs import utils

# base_url = utils.get_relative_url('.','.')
# print ('base_url',base_url)
root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
clean = True
errors = []
benchmark_dois = root_dir + "/benchmarks/benchmark_dois.json"
benchmark_dois = loadjson(benchmark_dois)
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
benchmark_descriptions = pd.read_csv(root_dir + "/benchmarks/descriptions.csv")
# TODO: Merge benchmark_dois and benchmark_descriptions


def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def make_summary_table():
    methods = ["AI", "ES", "FF", "QC", "EXP"]
    tasks = [
        "SinglePropertyPrediction",
        "SinglePropertyClass",
        "MLFF",
        "TextClass",
        "TokenClass",
        "TextSummary",
        "TextGen",
        "AtomGen",
        "ImageClass",
        "Spectra",
        "EigenSolver",
    ]
    line = '<table style="width:100%" id="j_table">'
    line += "<thead><td>Category/Sub-cat.</td>"
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
        elif "<h3>Artificial intelligence (AI)</h3><p>Contributions:" in j:
            with open("docs/AI/index.md", "r") as temp_file:
                temp_filedata = temp_file.read().splitlines()
                for k in temp_filedata:
                    if "<!--number_of_contributions-->" in k:
                        num_temp = int(k.split(":")[-1])
                        break
            temp = (
                "<h3>Artificial intelligence (AI)</h3><p>Contributions: "
                + str(num_temp)
                + "</p>"
            )
            content.append(temp)

        elif "<h3>Electronic Struct. (ES)</h3><p>Contributions:" in j:
            with open("docs/ES/index.md", "r") as temp_file:
                temp_filedata = temp_file.read().splitlines()
                for k in temp_filedata:
                    if "<!--number_of_contributions-->" in k:
                        num_temp = int(k.split(":")[-1])
                        break
            temp = (
                "<h3>Electronic Struct. (ES)</h3><p>Contributions: "
                + str(num_temp)
                + "</p>"
            )
            content.append(temp)

        elif "<h3>Force-field (FF)/potentials</h3><p>Contributions" in j:
            with open("docs/FF/index.md", "r") as temp_file:
                temp_filedata = temp_file.read().splitlines()
                for k in temp_filedata:
                    if "<!--number_of_contributions-->" in k:
                        num_temp = int(k.split(":")[-1])
                        break
            temp = (
                "<h3>Force-field (FF)/potentials</h3><p>Contributions "
                + str(num_temp)
                + "</p>"
            )
            content.append(temp)

        elif "<h3>Quantum Comput. (QC) </h3><p>Contributions:" in j:
            with open("docs/QC/index.md", "r") as temp_file:
                temp_filedata = temp_file.read().splitlines()
                for k in temp_filedata:
                    if "<!--number_of_contributions-->" in k:
                        num_temp = int(k.split(":")[-1])
                        break
            temp = (
                "<h3>Quantum Comput. (QC) </h3><p>Contributions: "
                + str(num_temp)
                + "</p>"
            )
            content.append(temp)
        elif "<h3>Experiments (EXP)</h3><p>Contributions:" in j:
            with open("docs/EXP/index.md", "r") as temp_file:
                temp_filedata = temp_file.read().splitlines()
                for k in temp_filedata:
                    if "<!--number_of_contributions-->" in k:
                        num_temp = int(k.split(":")[-1])
                        break
            temp = (
                "<h3>Experiments (EXP)</h3><p>Contributions: "
                + str(num_temp)
                + "</p>"
            )
            content.append(temp)

        elif "<h3>Contribution Guide</h3><p>Contributors:" in j:
            n_users = []
            try:
                rq = requests.get(
                    "https://api.github.com/repos/usnistgov/jarvis_leaderboard/contributors"
                ).json()
                for u in rq:
                    if u["login"] not in ["dependabot[bot]"]:
                        n_users.append(i)
            except Exception as exp:
                print("Cannot get users", exp)
                pass

            temp = (
                '<h3>Contribution Guide</h3><p>Contributors: <a href="https://github.com/usnistgov/jarvis_leaderboard/graphs/contributors" >'
                + str(len(n_users))
                + "</a></p>"
            )
            content.append(temp)

        elif "<h3>Methodologies</h3><p>Available Methods:" in j:
            n_methods = 0
            for n in glob.glob(
                "jarvis_leaderboard/contributions/*/metadata.json"
            ):
                # for i in glob.glob("jarvis_leaderboard/benchmarks/*/metadata.json"):
                n_methods += 1

            temp = (
                "<h3>Methodologies</h3><p>Available Methods:"
                + str(n_methods)
                + "</p>"
            )
            content.append(temp)

        else:
            content.append(j)
    # filedata = filedata.replace('<!--table_content-->', temp)

    with open(md_path, "w") as file:
        file.write("\n".join(content))


def get_metric_value(
    csv_path="../contributions/alignn_model/AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip",
    plot_filename=None,
):
    fname = csv_path.split("/")[-1].split(".csv.zip")[0]
    contribution = csv_path.split("/")[-2]
    temp = fname.split("-")
    category = temp[0]
    subcat = temp[1]
    prop = temp[2]
    dataset = temp[3]
    data_split = temp[4]
    metric = temp[-1]

    results = {}
    results["category"] = category
    results["subcat"] = subcat
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
        + contribution
    )  # meta_path.split('metadata.json')[0]

    # print("meta_path", meta_data)
    # meta_data=loadjson()
    # print("csv_data", csv_path)
    # dataset with actual values
    random_guessing_performance = "na"
    temp = dataset + "_" + prop + ".json"
    # print ('json temp',temp)
    temp2 = temp + ".zip"
    fname = os.path.join("benchmarks", category, subcat, temp2)
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
        errors.append([csv_path, len(csv_data), len(actual_df)])

    df = pd.merge(csv_data, actual_df, on="id")
    # print('csv',csv_path)
    # print ('df',df)
    # print('csv_data',csv_data)
    # print('actual_df',actual_df)
    results["res"] = "na"
    results["df"] = df

    if metric == "mae":
        res = round(mean_absolute_error(df["actual"], df["prediction"]), 4)
        # res = round(mean_absolute_error(df["actual"], df["prediction"]), 3)
        results["res"] = res
        if "qm9_std_jctc" in csv_path:
            # print('scaling[dataset][prop],',scaling[dataset][prop])
            res = round(
                scaling[dataset][prop]
                * mean_absolute_error(df["actual"], df["prediction"]),
                4,
            )
            results["res"] = res
            if plot_filename is not None:
                plt.plot(df["actual"], df["prediction"], ".")
                plt.savefig(plot_filename)
                plt.close()
            # print(csv_path)
            # print('mae1',mean_absolute_error(csv_data['target'],csv_data['prediction']))
            # print('res',res)
            # print(csv_data)
            # print(actual_df)
            # print()
    if metric == "acc":
        # print("ACC",csv_path)
        # print(df, len(df))
        res = round(accuracy_score(df["actual"], df["prediction"]), 4)
        # print("res", res)
        results["res"] = res
    if metric == "multimae":
        # print("csv multimae", csv_path)
        # print ('df',df)
        maes = []
        reals = []
        preds = []
        for k, v in df.iterrows():
            real = np.array(v["actual"].split(";"), dtype="float")
            # real = np.array(v["target"].split(";"), dtype="float")
            pred = np.array(v["prediction"].split(";"), dtype="float")
            # m = mean_absolute_error(real, pred)
            # maes.append(m)
            reals.append(real)
            preds.append(pred)
            # print('mm',m)
        # results["res"] = round(np.array(maes).sum() / len(maes), 4)
        results["res"] = mean_absolute_error(
            np.concatenate(reals), np.concatenate(preds)
        )
        if plot_filename is not None:
            plt.plot(np.concatenate(reals), np.concatenate(preds), ".")
            plt.savefig(plot_filename)
            plt.close()
        # print ('df',df)
        # print('csv_data',csv_data)
        # print('actual_df',actual_df)
        # print('res',results['res'])

    if metric == "mae" and "JVASP_" not in csv_path:
        if len(json_data["train"]) == 0:
            tdata = []
            for m in list(json_data["test"].values()):
                tdata.append(m)
        else:
            tdata = []
            for m in list(json_data["train"].values()):
                tdata.append(m)

        tdata = np.array(tdata, dtype="float")
        random_guessing_performance = mean_absolute_deviation(tdata)
        results["random_guessing_performance"] = random_guessing_performance
    if metric == "acc" and "JVASP_" not in csv_path:
        if len(json_data["train"]) == 0:
            tdata = []
            for m in list(json_data["test"].values()):
                tdata.append(m)
        else:
            tdata = []
            for m in list(json_data["train"].values()):
                tdata.append(m)

        random_guessing_performance = 1 / len(set(tdata))

        results["random_guessing_performance"] = random_guessing_performance
    if (
        metric == "multimae"
        and "JVASP_" not in csv_path
        and len(list(json_data["test"].values())) != 1
    ):
        # m=list(json_data["train"].values())[0].split(';')
        # print ('json_data["train"].values()',np.mean(np.array(m,dtype='float')))
        if len(json_data["train"]) == 0:
            tdata = []
            for m in list(json_data["test"].values()):
                for n in np.array(m.split(";"), dtype="float"):
                    tdata.append(n)
        else:
            tdata = []
            for m in list(json_data["train"].values()):
                for n in np.array(m.split(";"), dtype="float"):
                    tdata.append(n)
        tdata = np.array(tdata, dtype="float")
        avg = np.mean(tdata)
        random_guessing_performance = mean_absolute_error(
            tdata, np.repeat(avg, len(tdata))
        )
        results["random_guessing_performance"] = random_guessing_performance
    if metric == "rouge":
        # Due to dependency conflicts now avoiding evaluate package
        # import evaluate
        # from datasets import load_metric
        # metric = load_metric("rouge")
        # TODO: merge with benchmark instead of using target from csv.zip
        # rouge_score = evaluate.load("rouge")
        # scores = rouge_score.compute(
        #    predictions=df["prediction"], references=df["actual"]
        # )
        # scores = rouge_score.compute(predictions=csv_data['prediction'],references=csv_data['target'])
        # rouge = scores["rouge1"]
        # rouge=(calc_rouge_scores(df['target'],df['prediction']))['rouge1']
        # results["res"] = round(rouge, 4)
        # print('rouge eval',rouge)
        from rouge import Rouge

        rouge = Rouge()
        hypothesis = df["prediction"]
        reference = df["actual"]
        scores = rouge.get_scores(hypothesis, reference, avg=True)["rouge-1"][
            "r"
        ]
        results["res"] = round(scores, 4)
        print("rouge scores", scores)
    if metric == "rmse" and subcat == "AtomGen":
        print("AtomGen")
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        rms = []
        for m, mm in df.iterrows():
            try:
                atoms_target = (
                    Poscar.from_string(
                        (mm["actual"].replace("\\n", "\n"))
                    ).atoms
                ).pymatgen_converter()
                atoms_pred = (
                    Poscar.from_string(
                        (mm["prediction"].replace("\\n", "\n"))
                    ).atoms
                ).pymatgen_converter()
                # rms_dist = matcher.get_rms_dist(atoms_pred,atoms_target)
                rms_dist = matcher.get_rms_anonymous(atoms_pred, atoms_target)
                if rms_dist[0] is not None:
                    rms.append(rms_dist[0])
            except Exception as exp:
                print("exp", exp)
                pass
        try:
            rms = round(np.array(rms).mean(), 4)
        except:
            rms = -9999
            pass
        results["res"] = rms
        # import sys
        # sys.exit()
    return results


def check_metadata_json_exists():
    search = root_dir + "/contributions"
    all_dirs = []
    all_dirs_meta = []
    all_dirs_with_metadata = []
    for i in os.listdir(search):
        all_dirs.append(i)
        for j in os.listdir(os.path.join(search, i)):
            if "metadata.json" in j:
                meta_path = os.path.join(search, i, j)
                if meta_path not in all_dirs_with_metadata:
                    all_dirs_with_metadata.append(meta_path)
                    all_dirs_meta.append(i)
    problem_dirs = set(all_dirs) - set(all_dirs_meta)
    return problem_dirs


def check_metadata_info_exists():
    search = root_dir + "/contributions/*/" + "metadata.json"
    all_dirs = []
    all_dirs_ok = []
    for i in glob.glob(search):
        meta_data = loadjson(i)
        all_dirs.append(i)
        if (
            "author_email" in meta_data
            and "project_url" in meta_data
            and "git_url" in meta_data
            and "model_name" in meta_data
            and "@" in meta_data["author_email"]
            and "team_name" in meta_data
            and "time_taken_seconds" in meta_data
            and "software_used" in meta_data
            and "hardware_used" in meta_data
        ):
            all_dirs_ok.append(i)
    problem_dirs = set(all_dirs) - set(all_dirs_ok)
    return problem_dirs


def old_check_metadata_info_exists():
    search = root_dir + "/contributions/*/" + "metadata.json"
    all_dirs = []
    all_dirs_ok = []
    for i in glob.glob(search):
        meta_data = loadjson(i)
        all_dirs.append(i)
        if (
            "author_email" in meta_data
            and "project_url" in meta_data
            and "model_name" in meta_data
            and "@" in meta_data["author_email"]
            and "team_name" in meta_data
            and "time_taken_seconds" in meta_data
            and "software_used" in meta_data
            and "hadrware_used" in meta_data
        ):
            all_dirs_ok.append(i)
        # else:
        #    print(i)
        #    print(
        #        "Check metadata",
        #        i,
        #        meta_data["author_email"],
        #        meta_data["project_url"],
        #        meta_data["model_name"],
        #        meta_data["team_name"],
        #        meta_data["time_taken_seconds"],
        #        meta_data["software_used"],
        #        meta_data["hardware_used"],
        #    )
    problem_dirs = set(all_dirs) - set(all_dirs_ok)
    return problem_dirs


def check_run_sh_exists():
    search = root_dir + "/contributions"
    all_dirs = []
    all_dirs_meta = []
    all_dirs_with_metadata = []
    for i in os.listdir(search):
        all_dirs.append(i)
        for j in os.listdir(os.path.join(search, i)):
            if "run.sh" in j:
                meta_path = os.path.join(search, i, j)
                if meta_path not in all_dirs_with_metadata:
                    all_dirs_with_metadata.append(meta_path)
                    all_dirs_meta.append(i)
    problem_dirs = set(all_dirs) - set(all_dirs_meta)
    # for j in problem_dirs:
    #    p = os.path.join(search, j)
    #    runsh = os.path.join(p,'run.sh')
    #    f=open(runsh,'w')
    #    f.write('#!/bin/bash\n')
    #    for k in os.listdir(p):
    #      if '.py' in k:
    #          print(runsh,k)
    #          line='python '+k+'\n'
    #          f.write(line)
    #    f.close()
    return problem_dirs


def check_at_least_one_csv_zip_exists():
    search = root_dir + "/contributions"
    all_dirs = []
    all_dirs_meta = []
    all_dirs_with_metadata = []
    for i in os.listdir(search):
        tmp_path = os.path.join(search, i)
        if os.path.isdir(tmp_path):
            all_dirs.append(i)
            for j in os.listdir(tmp_path):
                if "csv.zip" in j:
                    meta_path = os.path.join(search, i, j)
                    if meta_path not in all_dirs_with_metadata:
                        all_dirs_with_metadata.append(meta_path)
                        all_dirs_meta.append(i)
    problem_dirs = set(all_dirs) - set(all_dirs_meta)
    return problem_dirs


def check_json_zip_exists_for_csv_zip():
    search = root_dir + "/contributions"
    # print('search',search)
    problem_csv = []
    for i in os.listdir(search):
        tmp_path = os.path.join(search, i)
        if os.path.isdir(tmp_path):
            for j in os.listdir(tmp_path):
                if "csv.zip" in j:
                    tmp = j.split("-")
                    # print(tmp)
                    bench_path = root_dir + "/benchmarks"
                    fname = tmp[3] + "_" + tmp[2] + ".json.zip"

                    json_path = os.path.join(bench_path, tmp[0], tmp[1], fname)

                    if not os.path.exists(json_path):
                        print(json_path)
                        problem_csv.append(json_path)

    # ES-Spectra-dielectric_function_JVASP_266_InP-dft_3d-test-multimae.csv.zip
    return problem_csv


def get_doi(
    bench_name="ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip",
):
    tmp = bench_name.split("-")
    cat = tmp[0]
    subcat = tmp[1]
    prop = tmp[2]
    dataset = tmp[3]
    json_zip_name = dataset + "_" + prop + ".json.zip"
    dois = benchmark_dois[cat][subcat][json_zip_name]
    return dois


def get_all_dois():
    all_dois = []
    search = root_dir + "/benchmarks/*/*/*.json.zip"
    for i in glob.glob(search):
        tmp = i.split("/")
        cat = tmp[-3]
        subcat = tmp[-2]
        bench = tmp[-1]
        dois = benchmark_dois[cat][subcat][bench]
        all_dois.append(dois)
    return all_dois


def get_results(
    bench_name="ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip",
    include_random=False,
    include_all_results=False,
):
    search = root_dir + "/contributions/*/" + bench_name
    vals = []
    names = []
    if include_all_results:
        results = []
    # for i in glob.glob("../contributions/*/AI-MLFF-forces-mlearn_Si-test-multimae.csv.zip"):
    for i in glob.glob(search):
        # print(i)
        res = get_metric_value(csv_path=i)
        # print (res['res'],res['random_guessing_performance'])
        if include_random:
            rand = res["random_guessing_performance"]
        model_name = res["model_name"]
        val = res["res"]
        vals.append(val)
        names.append(i.split("/")[-2])
        if include_all_results:
            results.append(res)
    if include_random:
        vals.append(rand)

    vals = np.array(vals)
    order = np.argsort(vals)
    vals = vals[order]
    if include_random:
        names.append("Random")
    names = np.array(names)
    names = names[order]
    if include_all_results:
        return names, vals, results
    return names, vals


def get_plotly(
    bench_name="ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip",
):
    # print('bench',bench_name)
    if "csv.zip" not in bench_name:
        bench_name = bench_name + ".csv.zip"
    names, vals, all_res = get_results(
        bench_name=bench_name, include_all_results=True
    )
    proj_urls = []
    for i in all_res:
        proj_urls.append(i["project_url"])
    # print('names',names)
    # print('vals',vals)
    nm_tmp = bench_name.split(".csv.zip")[0].split("-")
    plt_dat = [go.Bar(x=list(names), y=list(vals))]

    layout = {
        "yaxis": {"title": nm_tmp[-1].upper() + " (" + nm_tmp[2] + ")"},
        #'yaxis': {'title': 'Proportion (%)'},
        "title": {"text": bench_name.split(".csv.zip")[0]},
        "title_x": 0.5,
    }
    fig = go.Figure(data=plt_dat, layout=layout)
    tmp_out = str(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    return tmp_out.replace("\n", " ").replace("  ", " "), list(set(proj_urls))


def get_benchmark_description(
    bench_name="ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip",
    include_plot=True,
    include_doi=True,
):
    tmp = bench_name.split("-")
    cat = tmp[0]
    subcat = tmp[1]
    prop = tmp[2]
    dataset = tmp[3]
    proj_urls = []
    desc = (
        benchmark_descriptions[
            (benchmark_descriptions["Category"] == cat)
            & (benchmark_descriptions["Sub-category"] == subcat)
            & (benchmark_descriptions["Benchmark"] == dataset + "_" + prop)
        ]["Description"]
        .fillna("Coming soon")
        .values[0]
    )
    if include_plot:
        tmp_out, proj_urls = get_plotly(bench_name=bench_name)
        desc = desc + "<br>" + tmp_out
    if include_doi:
        tmp_dois = [
            str("[") + i + str("](") + i + ")"
            for i in list(set(get_doi(bench_name=bench_name) + proj_urls))
        ]
        doi_text = "Reference(s): " + ", ".join(tmp_dois) + "<br>"
        desc = desc + "<br>" + doi_text
    if desc == "NaN":
        print("NaN", bench_name)
        desc = ""
    return desc


def get_metric_value_old(
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
        # "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/"
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
    # fname = os.path.join("dataset", method, submod, temp2)
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
        res = round(mean_absolute_error(df["actual"], df["prediction"]), 4)
        results["res"] = res
        if "qm9_std_jctc" in csv_path:
            # print('scaling[dataset][prop],',scaling[dataset][prop])
            res = round(
                scaling[dataset][prop]
                * mean_absolute_error(df["actual"], df["prediction"]),
                4,
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
        res = round(accuracy_score(df["actual"], df["prediction"]), 4)
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
        results["res"] = round(np.array(maes).sum() / len(maes), 4)
        # results["res"] = round(np.array(maes).sum(), 3)
        # print ('df',df)
        # print('csv_data',csv_data)
        # print('actual_df',actual_df)
        # print('res',results['res'])

    return results


def rebuild_pages(
    exclude_benchs=["AI-AtomGen-heat_ref-perov5-test-rmse.csv.zip"],
    debug_one=False,
):
    print("Rebuilding web:")
    unique_fname = []
    os.chdir(root_dir + "/..")
    num_data = 0
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
        bnch_tmp = i.split("/")[-1]
        if bnch_tmp not in exclude_benchs:
            # for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
            # if 'Text' in i:
            print(i)
            fname = bnch_tmp.split(".csv.zip")[0]
            ###fname = i.split("/")[-1].split(".csv.zip")[0]
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
            md_filename = (
                "../docs/" + method + "/" + submod + "/" + prop + ".md"
            )
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
                elif "<!--benchmark_description-->" in j:
                    content.append("<!--benchmark_description-->")
                else:
                    content.append(j)
            with open(md_path, "w") as file:
                file.write("\n".join(content))
        if debug_one:
            break
    # jarvis_leaderboard/dataset/AI/dft_3d_exfoliation_energy.json
    dat = []
    md_files = []
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
        bnch_tmp = i.split("/")[-1]
        if bnch_tmp not in exclude_benchs:
            # for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
            print(i)
            fname = bnch_tmp.split(".csv.zip")[0]
            ###fname = i.split("/")[-1].split(".csv.zip")[0]
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

            category = temp[0]
            # method = temp[0]
            subcat = temp[1]
            # submod = temp[1]
            data_split = temp[4]
            prop = temp[2]
            dataset = temp[3]
            metric = temp[-1]

            team = i.split("/")[-2]
            md_filename = (
                "../docs/" + category + "/" + subcat + "/" + prop + ".md"
            )
            # md_filename = "../docs/" + method + "/" + submod + "/" + prop + ".md"
            md_filename = (
                "../docs/"
                + category
                # + method
                + "/"
                + subcat
                # + submod
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
            json_path = category + "/" + subcat + "/" + json_name
            # json_path = method + "/" + submod + "/" + json_name
            json_url = (
                '<a href="'
                + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/"
                # + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/dataset/"
                + json_path
                + '" target="_blank">JSON</a>'
            )
            metadata = (
                '<a href="'
                + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/"
                + "jarvis_leaderboard/contributions/"
                # + "jarvis_leaderboard/benchmarks/"
                + bench_name
                + "/metadata.json "
                + '" target="_blank">Info</a>'
            )
            runsh = (
                '<a href="'
                + "https://github.com/usnistgov/jarvis_leaderboard/tree/main/"
                + "jarvis_leaderboard/contributions/"
                # + "jarvis_leaderboard/benchmarks/"
                + bench_name
                + "/run.sh "
                + '" target="_blank">run.sh</a>'
            )
            notes = notes + ", " + json_url + ", " + runsh + ", " + metadata
            if "JVASP" in prop:
                jid = "JVASP-" + prop.split("JVASP_")[1].split("_")[0]
                # print("propjid", prop, jid)
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
                # submod=submod,
                csv_path=i,
                # dataset=dataset,
                # prop=prop,
                # data_split=data_split,
                # method=method,
                # metric=metric,
                # bench_name=bench_name,
            )
            # print ('res',res)
            if fname not in unique_fname:
                unique_fname.append(fname)
                num_data += res["dataset_size"]
            # num_data+=res['dataset_size']
            # num_data+=res['num_entries']
            print(
                "num_data",
                i,
                num_data,
                res["dataset_size"],
                res["num_entries"],
            )
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
                elif "<!--benchmark_description-->" in j:
                    bench_desc = get_benchmark_description(fname)
                    temp2 = (
                        "<!--benchmark_description--> - Description: "
                        + str(bench_desc)
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
        if debug_one:
            break
    # print("dat", dat)
    print("mdfiles", len(set(md_files)))
    print("exclude_benchs", len(exclude_benchs))

    def update_individual_index_md(
        md_path="docs/ES/index.md", key="ES", extra_key="-", homepage=[]
    ):
        n_methods = 0
        for i in glob.glob("jarvis_leaderboard/contributions/*/metadata.json"):
            # for i in glob.glob("jarvis_leaderboard/benchmarks/*/metadata.json"):
            n_methods += 1
        if not homepage:
            homepage = []
            for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
                # for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):

                if i.split("/")[-1].split("-")[0] == key and extra_key in i:
                    # if key in i and extra_key in i:
                    # TODO: Distinguish MLFF and FF
                    p = i.split("/")[-1].split(".csv.zip")[0]
                    homepage.append(p)
        # print ('index pages',homepage)
        # print("dat", dat)
        # print("errors", errors, len(errors))
        # Keep only selected on homepage
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
                    i["result"]["category"]
                    # i["result"]["method"]
                    + "-"
                    + i["result"]["subcat"]
                    # + i["result"]["submod"]
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
            + "<th>Category</th>"
            # + "<th>Method</th>"
            + "<th>Sub-category</th>"
            # + "<th>Task</th>"
            + "<th>Benchmark</th>"
            # + "<th>Property</th>"
            + "<th>Method</th>"
            # + "<th>Model name</th>"
            + "<th>Metric</th>"
            + "<th>Score</th>"
            + "<th>Team</th>"
            + "<th>Dataset</th>"
            + "<th>Size</th>"
            + "</tr></thead>"
        )
        for i, j in selected.items():
            if len(md_path.split("/")) == 2:
                # e.g.https://pages.nist.gov/jarvis_leaderboard/
                # if md_path == "docs/index.md":
                # desc = get_benchmark_description(i)
                # print('len 2',i,desc)
                # temp+=desc
                temp = (
                    temp
                    + "<tr>"
                    + "<td>"
                    # + '<a href="./'
                    + j["category"]
                    # + '" target="_blank">'
                    # + j["category"]
                    # + "</a>"
                    + "</td>"
                    + "<td>"
                    # + '<a href="./'
                    # + j["category"]
                    # + "/"
                    + j["subcat"]
                    # + '" target="_blank">'
                    # + j["subcat"]
                    # + "</a>"
                    + "</td>"
                    + "<td>"
                    + '<a href="./'
                    + j["category"]
                    + "/"
                    + j["subcat"]
                    + "/"
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + '" target="_blank">'
                    + j["dataset"]
                    + "_"
                    + j["prop"]
                    + "</a>"
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
                    + "</tr>"
                )
            elif len(md_path.split("/")) == 3:
                # e.g.https://pages.nist.gov/jarvis_leaderboard/EXP/
                # print('len 3',i)
                base = "."
                temp = (
                    temp
                    + "<tr>"
                    + "<td>"
                    # + '<a href= "'
                    # + base
                    # + "/"
                    + j["category"]
                    # + '" target="_blank">'
                    # + j["category"]
                    # + "</a>"
                    + "</td>"
                    + "<td>"
                    # + '<a href= "'
                    # + base
                    # + "/"
                    + j["subcat"]
                    # + '" target="_blank">'
                    # + j["subcat"]
                    # + "</a>"
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    + j["subcat"]
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
            elif len(md_path.split("/")) == 4:
                # print('len 4',i)
                # desc = get_benchmark_description(i)
                # print('len 4',i,desc)
                # temp+=desc
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
                    + j["category"]
                    # + j["method"]
                    + '" target="_blank">'
                    + j["category"]
                    # + j["method"]
                    + "</a>"
                    # + j["method"]
                    + "</td>"
                    + "<td>"
                    + '<a href= "'
                    + base
                    + "/"
                    # + '<a href="http://127.0.0.1:8000/knc6/jarvis_leaderboard/'
                    # + j["method"]
                    + j["category"]
                    + "/"
                    + j["subcat"]
                    # + j["submod"]
                    + '" target="_blank">'
                    # + j["submod"]
                    + j["subcat"]
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
            elif "<!--number_of_contributors-->" in j:
                content.append("<!--number_of_contributors-->")
            elif "<!--number_of_benchmarks-->" in j:
                content.append("<!--number_of_benchmarks-->")
            elif "<!--number_of_methods-->" in j:
                content.append("<!--number_of_methods-->")
            elif "<!--number_of_datapoints-->" in j:
                content.append("<!--number_of_datapoints-->")
            # elif "<!--benchmark_description-->" in j:
            #    content.append("<!--benchmark_description-->")
            #    print('benchmark_description-')
            #    import sys
            #    sys.exit()
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
            elif "<!--number_of_contributors-->" in j:
                n_users = []
                try:
                    rq = requests.get(
                        "https://api.github.com/repos/usnistgov/jarvis_leaderboard/contributors"
                    ).json()
                    for u in rq:
                        if u["login"] not in ["dependabot[bot]"]:
                            n_users.append(i)
                except Exception as exp:
                    print("Cannot get users", exp)
                    pass

                temp2 = (
                    "<!--number_of_contributors--> - Number of contributors: "
                    + "["
                    + str(len(n_users))
                    + "]"
                    + "(https://github.com/usnistgov/jarvis_leaderboard/graphs/contributors)"
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
        "AI-MLFF-forces-mlearn_Si-test-multimae",
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
    y = []
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
        # for i in glob.glob("jarvis_leaderboard/benchmarks/*/*.csv.zip"):
        x.append(i.split(".csv.zip")[0])
        tmp = i.split("/")[-1].split(".csv.zip")[0]
        if tmp not in y:
            y.append(tmp)
        # x.append(i.split('/')[-1].split('.csv.zip')[0])
    y = sorted(y)
    print("Files", len(x))
    # update_individual_index_md(md_path="docs/index.md", homepage=y)
    update_individual_index_md(md_path="docs/index.md", homepage=homepage)
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
    update_individual_index_md(
        md_path="docs/AI/TokenClass/index.md", key="AI", extra_key="TokenClass"
    )
    update_individual_index_md(
        md_path="docs/AI/TextSummary/index.md",
        key="AI",
        extra_key="TextSummary",
    )
    update_individual_index_md(
        md_path="docs/AI/TextGen/index.md", key="AI", extra_key="TextGen"
    )
    update_individual_index_md(
        md_path="docs/AI/AtomGen/index.md", key="AI", extra_key="AtomGen"
    )
    update_individual_index_md(md_path="docs/QC/index.md", key="QC")
    update_individual_index_md(
        md_path="docs/QC/EigenSolver/index.md",
        key="QC",
        extra_key="EigenSolver",
    )
    update_individual_index_md(md_path="docs/EXP/index.md", key="EXP")
    update_individual_index_md(
        md_path="docs/EXP/SinglePropertyPrediction/index.md",
        key="EXP",
        extra_key="SinglePropertyPrediction",
    )
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
    # x = check_json_zip_exists_for_csv_zip()
    # d = get_all_dois()
    # print(len(d))
