import glob
from jarvis_leaderboard.rebuild import get_metric_value, get_results
import pprint
from collections import defaultdict
import numpy as np
import plotly.express as px
import pandas as pd
import os

root_dir = os.path.dirname(os.path.abspath(__file__))


def catalysis_mat(
    benchmarks=[
        "AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-tinnet_O-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-tinnet_OH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_O-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_OH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_CO-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_COOH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_CHO-test-mae.csv.zip",
    ],
    metric="pearsonr",
    replacements=[
        "Out-",
        "AI-SinglePropertyPrediction-ead-",
        "-test-mae.csv.zip",
    ],
    md_path="catalysis_mat.md",
    width=800,
    height=800,
    desired_order=[],
):
    # Find all methods that have contributions for the above benchmarks
    mem = {}
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
        for j in benchmarks:
            if j in i:
                tmp = i.split("/")[2]
                for r in replacements:
                    tmp = tmp.replace(r, "")
                mem[tmp] = {}
                for k in benchmarks:
                    mem[tmp][k] = []

    for i in benchmarks:
        names, vals = get_results(bench_name=i, metric=metric)
        for j, k in zip(names, vals):
            for r in replacements:
                j = j.replace(r, "")
            mem[j][i] = float(k)

    dat = []
    row_names = []
    for i, j in mem.items():
        row_names.append(i)
        tmp = []
        for m, n in j.items():
            tmp.append(n)
        dat.append(tmp)
    column_names = []
    for b in benchmarks:
        for r in replacements:
            b = b.replace(r, "")
        column_names.append(b)
    print(column_names)
    df = pd.DataFrame(dat, index=row_names, columns=column_names)
    # print(df)
    if not desired_order:
        desired_order = column_names
    # Reindex the DataFrame to have both rows and columns in the desired order
    df_reordered = df.reindex(index=desired_order, columns=column_names)

    # Display the reordered DataFrame
    print(df_reordered)
    # fig = px.imshow(df, text_auto=True)
    fig = px.imshow(df_reordered, text_auto=True)
    fig.update_layout(
        width=width,  # Set the width of the figure
        height=height,  # Set the height of the figure
    )
    htm = fig.to_html()
    tmp_out = str(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    tmp_out = tmp_out.replace("\n", " ").replace("  ", " ")
    md_path = os.path.join(root_dir, "..", "docs", "Special", md_path)
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
    filedata = "\n".join(content)
    filedata = filedata.replace(
        "<!--table_content-->", "<!--table_content-->" + tmp_out
    )
    with open(md_path, "w") as file:
        file.write(filedata)


def chips_ff(
    benchmarks=[
        "AI-SinglePropertyPrediction-a-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-b-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c11-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c44-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-form_en-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-kv-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-vol-dft_3d_chipsff-test-mae.csv.zip",
    ],
    metric="pearsonr",
    replacements=[
        "AI-SinglePropertyPrediction-",
        "-test-mae.csv.zip",
    ],
    md_path="CHIPS_FF.md",
    width=800,
    height=800,
    desired_order=[],
):
    # Find all methods that have contributions for the above benchmarks
    mem = {}
    for i in glob.glob("jarvis_leaderboard/contributions/*/*.csv.zip"):
        for j in benchmarks:
            if j in i:
                tmp = i.split("/")[2]
                for r in replacements:
                    tmp = tmp.replace(r, "")
                mem[tmp] = {}
                for k in benchmarks:
                    mem[tmp][k] = []

    for i in benchmarks:
        names, vals = get_results(bench_name=i, metric=metric)
        for j, k in zip(names, vals):
            for r in replacements:
                j = j.replace(r, "")
            mem[j][i] = float(k)
    # print('mem',mem)
    dat = []
    row_names = []
    for i, j in mem.items():
        row_names.append(i)
        tmp = []
        for m, n in j.items():
            tmp.append(n)
        dat.append(tmp)
    column_names = []
    for b in benchmarks:
        for r in replacements:
            b = b.replace(r, "")
        column_names.append(b)
    # print("column names",column_names)
    df = pd.DataFrame(dat, index=row_names, columns=column_names)
    # print('df',df)
    if not desired_order:
        desired_order = column_names
    # Reindex the DataFrame to have both rows and columns in the desired order
    df_reordered = df  # df.reindex(index=desired_order, columns=column_names)

    # Display the reordered DataFrame
    # print(df_reordered)
    # fig = px.imshow(df, text_auto=True)
    fig = px.imshow(df_reordered, text_auto=True)
    fig.update_layout(
        width=width,  # Set the width of the figure
        height=height,  # Set the height of the figure
    )
    htm = fig.to_html()
    tmp_out = str(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    tmp_out = tmp_out.replace("\n", " ").replace("  ", " ")
    md_path = os.path.join(root_dir, "..", "docs", "Special", md_path)
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
    filedata = "\n".join(content)
    filedata = filedata.replace(
        "<!--table_content-->", "<!--table_content-->" + tmp_out
    )
    with open(md_path, "w") as file:
        file.write(filedata)


if __name__ == "__main__":
    htm = chips_ff()
    # print(htm)
    htm = catalysis_mat(
        desired_order=[
            "tinnet_N",
            "tinnet_O",
            "tinnet_OH",
            "AGRA_O",
            "AGRA_OH",
            "AGRA_CO",
            "AGRA_COOH",
            "AGRA_CHO",
            "alignnff_pretrained_wt0.1",
            "CHGNet_pretrained",
            "MACE_pretrained",
            "MATGL_pretrained",
        ]
    )
