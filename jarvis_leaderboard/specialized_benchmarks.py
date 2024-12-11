import os
import glob
import pandas as pd
import plotly.express as px
from jarvis_leaderboard.rebuild import get_results

root_dir = os.path.dirname(os.path.abspath(__file__))


def process_benchmarks(
    benchmarks,
    metric,
    replacements,
    md_path,
    width=800,
    height=800,
    desired_order=None,
    add_links=True,
):
    mem = {}
    # Collect contributions
    for contrib_path in glob.glob(
        "jarvis_leaderboard/contributions/*/*.csv.zip"
    ):
        for benchmark in benchmarks:
            if benchmark in contrib_path:
                method = contrib_path.split("/")[2]
                for r in replacements:
                    method = method.replace(r, "")
                mem.setdefault(method, {k: [] for k in benchmarks})

    # Populate benchmark results
    for benchmark in benchmarks:
        names, vals = get_results(bench_name=benchmark, metric=metric)
        for name, val in zip(names, vals):
            for r in replacements:
                name = name.replace(r, "")
            mem[name][benchmark] = float(val)
    detailed_links = (
        '<table style="width:100%" id="j_table">'
        + "<thead><tr>"
        + "<th>Names</th>"
        + "<th>Links</th>"
        + "</tr></thead>"
    )
    if add_links:
        for benchmark in benchmarks:
            # print('benchmark',benchmark)
            temp = benchmark.split("-")
            category = temp[0]
            subcat = temp[1]
            data_split = temp[4]
            prop = temp[2]
            dataset = temp[3]
            metric = temp[-1]

            md_filename = (
                "https://pages.nist.gov/jarvis_leaderboard/"
                # "../docs/"
                + category
                # + method
                + "/"
                + subcat
                # + submod
                + "/"
                + dataset
                + "_"
                + prop
            )
            # print(md_filename)
            md_filename = (
                '<a href="'
                + md_filename
                + '" target="_blank">'
                + md_filename
                + "</a>"
            )
            # detailed_links.append([dataset+"-"+prop,md_filename])
            elem = (
                "<tr> "
                + "<td>"
                + dataset
                + "-"
                + prop
                + "</td>"
                + "<td>"
                + md_filename
                + "</td>"
                + "</tr>"
            )
            detailed_links += elem
        detailed_links += "</table>"
    # Prepare DataFrame
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
    # print(column_names)
    # row_names, dat = zip(*[(k, list(v.values())) for k, v in mem.items()])
    # column_names = [b.replace(r, "") for b in benchmarks for r in replacements]
    df = pd.DataFrame(dat, index=row_names, columns=column_names)

    # Reorder DataFrame
    # print('column_names',df.columns)
    # print('row_names',row_names)
    # print('column_names1',column_names)
    # print('desired_order',desired_order)
    # desired_order = desired_order or column_names
    if desired_order:

        df_reordered = df.reindex(index=desired_order, columns=column_names)
    else:

        df_reordered = (
            df  # .reindex(index=desired_order)#, columns=column_names)
        )
    # print('df_reordered',df_reordered)
    # Plot and save
    fig = px.imshow(df_reordered, text_auto=True)
    fig.update_layout(width=width, height=height)
    save_md(fig=fig, md_path=md_path, detailed_links=detailed_links)


def save_md(fig=None, md_path=None, detailed_links=None):
    md_path = os.path.join(root_dir, "..", "docs", "Special", md_path)
    tmp_out = (
        fig.to_html(full_html=False, include_plotlyjs="cdn")
        .replace("\n", " ")
        .replace("  ", " ")
    )

    with open(md_path, "r") as file:
        filedata = file.read().splitlines()
    content = []
    for j in filedata:
        if "<!--table_content-->" in j:
            content.append("<!--table_content-->")
        elif "<!--table_details-->" in j:
            content.append("<!--table_details-->")
        elif "<!--benchmark_description-->" in j:
            content.append("<!--benchmark_description-->")
        else:
            content.append(j)
    filedata = "\n".join(content)
    filedata = filedata.replace(
        "<!--table_content-->", "<!--table_content-->" + tmp_out
    )
    if detailed_links:
        filedata = filedata.replace(
            "<!--table_details-->", "<!--table_details-->" + detailed_links
        )

    with open(md_path, "w") as file:
        file.write(filedata)

    # with open(md_path, "r") as file:
    #    content = [
    #        (
    #            line
    #            if "<!--table_content-->" not in line
    #            else f"<!--table_content-->{tmp_out}"
    #        )
    #        for line in file
    #    ]
    # with open(md_path, "w") as file:
    #    file.write("\n".join(content))


if __name__ == "__main__":
    catalysis_benchmarks = [
        "AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-tinnet_O-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-tinnet_OH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_O-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_OH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_CO-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_COOH-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-ead-AGRA_CHO-test-mae.csv.zip",
    ]
    chips_benchmarks = [
        "AI-SinglePropertyPrediction-a-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-b-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-vol-dft_3d_chipsff-test-mae.csv.zip",
        #"AI-SinglePropertyPrediction-form_en-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c11-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-c44-dft_3d_chipsff-test-mae.csv.zip",
        "AI-SinglePropertyPrediction-kv-dft_3d_chipsff-test-mae.csv.zip",
        #"AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip",
        #"AI-SinglePropertyPrediction-vac_en-dft_3d_chipsff-test-mae.csv.zip",
    ]

    process_benchmarks(
        benchmarks=catalysis_benchmarks,
        metric="pearsonr",
        replacements=[
            "Out-",
            "AI-SinglePropertyPrediction-ead-",
            "-test-mae.csv.zip",
        ],
        md_path="catalysis_mat.md",
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
        ],
    )
    process_benchmarks(
        benchmarks=chips_benchmarks,
        metric="pearsonr",
        replacements=["AI-SinglePropertyPrediction-", "-test-mae.csv.zip"],
        md_path="CHIPS_FF.md",
    )
