import glob, json
from collections import defaultdict
from jarvis.db.jsonutils import dumpjson

# ../benchmarks/ES/SinglePropertyPrediction/dft_3d_Tc_supercon_JVASP_14837_V.json.zip
x = []
mem = defaultdict(lambda: defaultdict(dict))
for i in glob.glob("../benchmarks/*/*/*.json.zip"):
    tmp = i.split("/")
    cat = tmp[2]
    subcat = tmp[3]
    bench = tmp[4]
    default_doi = []
    if "dft_3d" in bench:
        default_doi = [
            "https://www.nature.com/articles/s41524-020-00440-1",
            "https://doi.org/10.48550/arXiv.2305.11842",
        ]
    if "dft_2d" in bench:
        default_doi = [
            "https://www.nature.com/articles/s41524-020-00440-1",
            "https://doi.org/10.48550/arXiv.2305.11842",
        ]
    if "halide_peroskites" in bench:
        default_doi = ["https://doi.org/10.1039/D1EE02971A"]
    if "mlearn" in bench:
        default_doi = ["https://doi.org/10.1021/acs.jpca.9b08723"]
    if "mlearn" in bench:
        default_doi = ["https://doi.org/10.1021/acs.jpca.9b08723"]
    if "stem_2d_image_bravais_class" in bench:
        default_doi = ["https://doi.org/10.1021/acs.jcim.2c01533"]
    if "alignn_ff_db" in bench:
        default_doi = ["https://doi.org/10.1039/D2DD00096B"]
    if "qm9" in bench:
        default_doi = ["https://doi.org/10.1038/sdata.2014.22"]
    if "vacancydb" in bench:
        default_doi = ["https://doi.org/10.48550/arXiv.2205.08366"]
    if "biobench" in bench:
        default_doi = [
            "https://doi.org/10.1021/acs.jctc.1c00431",
            "https://doi.org/10.1021/acs.jctc.2c00058",
        ]
    if "qe_tb" in bench:
        default_doi = ["https://doi.org/10.1103/PhysRevMaterials.7.044603"]
    if "megnet" in bench:
        default_doi = ["https://doi.org/10.1063/1.4812323"]
    if "m3gnet" in bench:
        default_doi = ["https://doi.org/10.1063/1.4812323"]
    if "hmof" in bench:
        default_doi = ["https://doi.org/10.1021/acs.jced.2c00583"]
    if "qmof" in bench:
        default_doi = ["https://doi.org/10.1016/j.matt.2021.02.015"]
    if "foundry_ml_exp_bandgaps" in bench:
        default_doi = ["https://doi.org/10.18126/wg3u-g8vu"]
    if "snumat" in bench:
        default_doi = ["https://doi.org/10.1038/s41597-020-00723-8"]
    if "snumat" in bench:
        default_doi = ["https://doi.org/10.1038/s41597-020-00723-8"]
    if "ocp" in bench:
        default_doi = ["https://doi.org/10.1021/acscatal.0c04525"]
    if "tinnet" in bench:
        default_doi = ["https://doi.org/10.1038/s41467-021-25639-8"]
    if "mag2d_chem" in bench:
        default_doi = ["https://doi.org/10.1038/s41598-020-72811-z"]
    if "supercon_chem" in bench:
        default_doi = ["https://doi.org/10.1038/s41524-018-0085-8"]
    if "edos_pdos" in bench:
        default_doi = ["https://doi.org/10.1103/PhysRevMaterials.7.023803"]
    if "nist_isodb" in bench:
        default_doi = ["https://doi.org/10.1007/s10450-018-9958-x"]
    if "arxiv" in bench or "arXiv" in bench:
        default_doi = ["https://doi.org/10.48550/arXiv.1905.00075"]
    if "pubchem" in bench:
        default_doi = ["https://doi.org/10.1093/nar/gkaa971"]
    if "mat_scholar" in bench:
        default_doi = ["https://doi.org/10.1021/acs.jcim.9b00470"]
    if "ssub" in bench:
        default_doi = ["https://doi.org/10.1007/978-3-540-45280-5_1"]
    if "mxene275" in bench:
        default_doi = ["https://doi.org/10.1088/2053-1583/ac1059"]
    if "midas_stress_strain" in bench:
        default_doi = ["https://doi.org/10.1177/0040517520918232"]
    if "lj_2d" in bench:
        default_doi = ["https://doi.org/10.1103/PhysRevA.45.5793"]
    mem[cat][subcat][bench] = default_doi
    # print (i,tmp)
    # cat=
print(json.dumps(mem, indent=4))

y = []
missing = []
for i, j in mem.items():
    for k, v in j.items():
        for m, n in v.items():
            if not n:
                print(i, k, m, n)
                missing.append(m)
print(missing, len(missing))
f = open("benchmark_dois.json", "w")
f.write(json.dumps(mem, indent=4))
f.close()
# dumpjson(data=mem,filename='benchmark_dois.json')
