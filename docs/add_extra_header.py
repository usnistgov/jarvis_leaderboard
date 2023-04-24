import glob

# for i in glob.glob('QC/EigenSolver/electron_bands.md'):
for i in glob.glob("*/*/*.md"):
    print(i)
    with open(i, "r") as file:
        filedata = file.read().splitlines()
    content = []
    for j in filedata:
        if "<th>Model name</th>" in j and "Dataset" not in j:
            temp = "<th>Model name</th><th>Dataset</th>"  # +"<!-- <th>Method</th>-->" + j
            content.append(temp)
        else:
            content.append(j)
    # filedata = filedata.replace('<!--table_content-->', temp)

    with open(i, "w") as file:
        file.write("\n".join(content))
