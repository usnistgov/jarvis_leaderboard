![Leaderboard actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/test_build.yml/badge.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/usnistgov/jarvis_leaderboard)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Upload_benchmark_to_jarvis_leaderboard.ipynb)


# JARVIS-Leaderboard

This project provides benchmark-performances of various methods for materials science applications using the datasets available in JARVIS-Tools databases. Some of the methods are: Artificial Intelligence (AI), Electronic Structure (ES), Force-field (FF), Qunatum Computation (QC) and Experiments (EXP). There are a variety of properties included in the benchmark. In addition to prediction results, we attempt to capture the underlyig software, hardware and instrumental frameworks to enhance reproducibility. This project is a part of the NIST-JARVIS infrastructure.

Website: https://pages.nist.gov/jarvis_leaderboard/


# Directory tree structure preview
```
├── jarvis_leaderboard
│   ├── dataset
│   │   ├── AI
│   │   │   ├── ImageClass
│   │   │   │   └── stem_2d_image_bravais_class.json.zip
│   │   │   ├── MLFF
│   │   │   │   ├── alignn_ff_db_energy.json.zip
│   │   │   │   └── prepare.py
│   │   │   ├── SinglePropertyClass
│   │   │   │   ├── dft_3d_magmom_oszicar.json.zip
│   │   │   │   └── ...
│   │   │   ├── SinglePropertyPrediction
│   │   │   │   ├── dft_3d_exfoliation_energy.json.zip
│   │   │   │   ├── dft_3d_formation_energy_peratom.json.zip
│   │   │   │   ├── ...
│   │   │   └── TextClass
│   │   │       ├── arXiv_categories.json.zip
│   │   │       └── pubchem_categories.json.zip
│   │   ├── ES
│   │   │   ├── SinglePropertyPrediction
│   │   │   │   ├── dft_3d_Tc_supercon_JVASP_1151_MgB2.json.zip
│   │   │   │   ├── ...
│   │   │   │   └── prepare.py
│   │   │   └── Spectra
│   │   │       ├── dft_3d_dielectric_function.json.zip
│   │   │       ├── ...
│   │   ├── EXP
│   │   │   └── Spectra
│   │   │       └── dft_3d_XRD_JVASP_19821_MgB2.json.zip
│   │   ├── FF
│   │   │   └── SinglePropertyPrediction
│   │   │       └── dft_3d_bulk_modulus_JVASP_867_Cu.json.zip
│   │   └── QC
│   │       └── EigenSolver
│   │           └── dft_3d_electron_bands_JVASP_816_Al_WTBH.json.zip
│   ├── benchmarks
│   │   ├── alignn_model
│   │   │   ├── AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv.zip
│   │   │   ├── AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip
│   │   │   ├── AI-Spectra-ph_dos-dft_3d-test-multimae.csv.zip
│   │   │   ├── ...
│   │   │   ├── metadata.json
│   │   │   ├── run.py
│   │   │   ├── run.sh
│   │   ├── densenet_model
│   │   │   ├── AI-ImageClass-bravais_class-stem_2d_image-test-acc.csv.zip
│   │   │   ├── metadata.json
│   │   │   └── run.sh
│   │   ├── qe_pbesol_gbrv
│   │   │   ├── ES-SinglePropertyPrediction-Tc_supercon_JVASP_1151_MgB2-dft_3d-test-mae.csv.zip
│   │   │   └── metadata.json
│   │   │   └── run.sh
│   │   ├── qiskit_vqd_SU2
│   │   │   ├── QC-EigenSolver-electron_bands_JVASP_816_Al_WTBH-dft_3d-test-multimae.csv.zip
│   │   │   └── metadata.json
│   │   ├── qmcpack_dmc_pbe
│   │   │   ├── ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip
│   │   │   ├── ...
│   │   │   ├── metadata.json
│   │   │   ├── run.py
│   │   │   └── run.sh
│   │   │   └── run.sh
│   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── ...
├── docs
│   ├── AI
│   │   ├── ImageClass
│   │   │   ├── bravais_class.md
│   │   │   └── index.md
│   │   ├── MLFF
│   │   │   ├── energy.md
│   │   │   └── index.md
│   │   ├── SinglePropertyClass
│   │   │   ├── index.md
│   │   │   └── ...
│   │   ├── SinglePropertyPrediction
│   │   │   ├── Band_gap_HSE.md
│   │   │   ├── exfoliation_energy.md
│   │   │   ├── formation_energy_peratom.md
│   │   │   ├── optb88vdw_bandgap.md
│   │   │   └── ...
│   │   ├── Spectra
│   │   │   ├── index.md
│   │   │   └── ph_dos.md
│   │   ├── TextClass
│   │   │   ├── categories.md
│   │   │   └── index.md
│   │   └── index.md
│   ├── ES
│   │   ├── SinglePropertyPrediction
│   │   │   ├── Tc_supercon_JVASP_1151_MgB2.md
│   │   │   ├── Tc_supercon_JVASP_11981_Nb3Al.md
│   │   │   └── ...
│   ├── populate_data.py
│   ├── rebuild.py
│   └── scripts
│       ├── convert.py
│       ├── format_data.py
│       └── transform.py
├── mkdocs.yml
├── requirements.txt
└── setup.py


```


# Citation

```
@article{choudhary2020joint,
  title={The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design},
  author={Choudhary, Kamal and Garrity, Kevin F and Reid, Andrew CE and DeCost, Brian and Biacchi, Adam J and Hight Walker, Angela R and Trautt, Zachary and Hattrick-Simpers, Jason and Kusne, A Gilad and Centrone, Andrea and others},
  journal={npj computational materials},
  volume={6},
  number={1},
  pages={173},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
```
