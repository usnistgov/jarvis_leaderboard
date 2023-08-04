x = {
    "doi": {
        "all": [
            "https://www.nature.com/articles/s41524-020-00440-1",
            "https://doi.org/10.48550/arXiv.2305.11842",
        ]
    }
}
import glob, json, zipfile

for i in glob.glob("*.json.zip"):
    if "dft_3d" in i:
        print(i)
