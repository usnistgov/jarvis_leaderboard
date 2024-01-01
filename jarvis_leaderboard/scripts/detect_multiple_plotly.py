import glob

# window.PLOTLYENV=window.PLOTLYENV
for i in glob.glob("docs/*/*/*.md"):
    with open(i, "r") as file:
        filedata = file.read().splitlines()
    nplotly = 0
    content = []
    for j in filedata:
        if "window.PLOTLYENV=window.PLOTLYENV" in j and nplotly == 0:
            content.append(j)
            nplotly += 1
        elif "window.PLOTLYENV=window.PLOTLYENV" in j and nplotly > 0:
            continue
        else:
            content.append(j)
    with open(i, "w") as file:
        file.write("\n".join(content))
