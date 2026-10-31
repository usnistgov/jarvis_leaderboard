# Magnetic classification with 772 structure descriptors and XGBoost

This contribution contains predictions for the JARVIS `AI/SinglePropertyClass/dft_3d_magmom_oszicar` test split. The submitted classifier averages three independently seeded XGBoost models and applies the fixed threshold 0.475.

The 772 descriptors comprise independently reconstructed composition/geometry features plus magnetic-element and local-environment descriptors. Inputs are JARVIS atomic structures and fixed elemental properties. Material DFT target values and other computed material properties are not model features.

The 0.475 threshold and candidate configuration were chosen using grouped training-internal splits. The XGBoost family was selected after comparing candidate families on the development validation set (95.6713% on 5,221 validation IDs after removing one train/validation duplicate). This is a development score, not the test score. The official test labels were not used to fit, select or adjust the model.

To reproduce the submitted predictions with the included weights, install `requirements.txt` under Python 3.12 and provide the official JARVIS-DFT 3D raw JSON and official benchmark split ZIP:

```sh
export JARVIS_RAW=/path/to/jdft_3d-12-12-2022.json
export JARVIS_SPLIT=/path/to/dft_3d_magmom_oszicar.json.zip
bash run.sh
```

`run.py` reads the test map keys for the ordered IDs, uses only each record's `jid` and `atoms` fields, verifies each weight's SHA-256, and writes the benchmark-named CSV and ZIP. It never fits a model or changes the threshold. The regenerated CSV has been verified to match the submitted CSV byte-for-byte in the recorded local environment. The three fixed model weights and the feature code are included for independent inference. The `training/` directory records the previously frozen search implementation and its configuration.

The original teammate feature package was unavailable to this account. The earlier 434 features were rebuilt independently; their definitions are documented in the feature-generation scripts rather than claimed to be byte-identical to the teammate's package.
