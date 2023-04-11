![GitHub repo size](https://img.shields.io/github/repo-size/usnistgov/jarvis_leaderboard)

# JARVIS-Leaderboard



This project benchmarks the performances of various methods for materials science applications using the datasets available in JARVIS-Tools databases.

Website: https://pages.nist.gov/jarvis_leaderboard/


# Directory tree structure preview
```
├── jarvis_leaderboard
├── docs
│   ├── AI
│   │   ├── ImageClass
│   │   │   ├── bravais_class.md
│   │   ├── MLFF
│   │   │   ├── energy.md
│   │   ├── SinglePropertyClass
│   │   │   ├── magmom_oszicar.md
│   │   │   ├── mbj_bandgap.md
│   │   │   ├── ...
│   │   ├── SinglePropertyPrediction
│   │   │   ├── formation_energy_peratom.md
│   │   │   ├── Band_gap_HSE.md
│   │   │   ├── HOMO.md
│   │   │   ├── formula_energy.md
│   │   │   └── ...
│   │   ├── Spectra
│   │   │   └── ph_dos.md
│   │   ├── TextClass
│   │   │   ├── categories.md
│   │   └── index.md
│   ├── ES
│   │   ├── SinglePropertyPrediction
│   │   │   ├── Tc_supercon_JVASP_1151_MgB2.md
│   │   │   ├── bandgap_JVASP_1002_Si.md
│   │   │   ├── epsx_JVASP_182_SiC.md
│   │   │   ├── slme_JVASP_7757_CdTe.md
│   │   │   └── ...
│   │   ├── Spectra
│   │   │   ├── dielectric_function_JVASP_1174_GaAs.md
│   │   │   └── ...
│   ├── EXP
│   │   ├── Spectra
│   │   │   ├── XRD_JVASP_19821_MgB2.md
│   ├── FF
│   │   ├── SinglePropertyPrediction
│   │   │   ├── bulk_modulus_JVASP_1002_Si.md
│   ├── QC
│   │   ├── EigenSolver
│   │   │   ├── electron_bands_JVASP_816_Al_WTBH.md
│   ├── benchmarks
│   │   ├── alignn_model
│   │   │   ├── SinglePropertyPrediction-test-surface_area_m2g-hmof-AI-mae.csv.zip
│   │   │   ├── SinglePropertyPrediction-test-void_fraction-hmof-AI-mae.csv.zip
│   │   │   ├── Spectra-test-ph_dos-dft_3d-AI-multimae.csv.zip
│   │   │   ├── run.sh
│   │   │   ├── metadata.json
│   │   │   ├── run.py
│   │   ├── densenet_model
│   │   │   ├── ImageClass-test-bravais_class-stem_2d_image-AI-acc.csv.zip
│   │   │   ├── ...
│   │   ├── vasp_optb88vdw
│   │   │   └── ...
│   ├── dataset
│   │   ├── AI
│   │   │   ├── ImageClass
│   │   │   │   └── stem_2d_image_bravais_class.json.zip
│   │   │   ├── MLFF
│   │   │   │   ├── alignn_ff_db_energy.json.zip
│   │   │   │   └── prepare.py
│   │   │   ├── SinglePropertyClass
│   │   │   │   ├── dft_3d_magmom_oszicar.json.zip
│   │   │   │   ├── dft_3d_mbj_bandgap.json.zip
│   │   │   │   └── ...
│   │   │   ├── SinglePropertyPrediction
│   │   │   │   ├── dft_3d_avg_elec_mass.json.zip
│   │   │   │   ├── dft_3d_avg_hole_mass.json.zip
│   │   │   │   └── ...
│   │   │   ├── Spectra
│   │   │   │   └── dft_3d_ph_dos.json.zip
│   │   │   └── TextClass
│   │   │       ├── arXiv_categories.json.zip
│   │   │       └── pubchem_categories.json.zip
│   │   ├── ES
│   │   │   ├── SinglePropertyPrediction
│   │   │   │   ├── dft_3d_Tc_supercon.json.zip
│   │   │   │   ├── dft_3d_Tc_supercon_JVASP_1014_Ta.json.zip
│   │   │   └── Spectra
│   │   │       ├── dft_3d_dielectric_function.json.zip
│   │   │       └── dft_3d_dielectric_function_JVASP_890_Ge.json.zip
│   │   ├── EXP
│   │   │   └── Spectra
│   │   │       └── dft_3d_XRD_JVASP_19821_MgB2.json.zip
│   │   ├── FF
│   │   │   └── SinglePropertyPrediction
│   │   │       ├── dft_3d_bulk_modulus_JVASP_1002_Si.json.zip
│   │   │       ├── dft_3d_bulk_modulus_JVASP_816_Al.json.zip
│   │   │       └── dft_3d_bulk_modulus_JVASP_867_Cu.json.zip
│   │   └── QC
│   │       └── EigenSolver
│   │           └── dft_3d_electron_bands_JVASP_816_Al_WTBH.json.zip
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
