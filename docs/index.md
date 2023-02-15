[LICENSE]: https://github.com/usnistgov/jarvis/blob/master/LICENSE.rst

# JARVIS Leaderboard [WIP]

This project provides benchmark-performances of various methods for materials science applications using the datasets available in [JARVIS-Tools databases](https://jarvis-tools.readthedocs.io/en/master/databases.html). Some of the methods are: [Artificial Intelligence (AI)](./AI), [Electronic Structure (ES)](./ES) and [Qunatum Computation (QC](./QC)). There are a variety of properties included in the benchmark.
In addition to prediction results, we attempt to capture the underlyig software and hardware frameworks in training models to enhance reproducibility. This project is a part of the [NIST-JARVIS](https://jarvis.nist.gov) infrastructure.


<!--number_of_benchmarks--> - Number of benchmarks: 89
- [Learn how to add benchmarks below](#add)
<!-- <p style="text-align:center;"><img align="middle" src="https://www.ctcms.nist.gov/~knc6/images/logo/jarvis-mission.png"  width="40%" height="20%"></p>-->


# Examples of benchmarks
<!--table_content--><table style="width:100%" id="j_table"><thead><tr><th>Method</th><th>Task</th><th>Property</th><th>Model name</th><th>Metric</th><th>Score</th><th>Team</th><th>Dataset</th><th>Size</th></tr></thead><tr><td><a href="./AI" target="_blank">AI</a></td><td><a href="./AI/SinglePropertyPrediction" target="_blank">SinglePropertyPrediction</a></td><td><a href="./AI/SinglePropertyPrediction/formation_energy_peratom" target="_blank">formation_energy_peratom</a></td><td><a href="https://www.nature.com/articles/s41524-021-00650-1" target="_blank">alignn_model</a></td><td>MAE</td><td>0.033</td><td>JARVIS</td><td>dft_3d</td><td>55713</td></tr><tr><td><a href="./AI" target="_blank">AI</a></td><td><a href="./AI/MLFF" target="_blank">MLFF</a></td><td><a href="./AI/MLFF/energy" target="_blank">energy</a></td><td><a href="https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b" target="_blank">alignnff_wt0.1</a></td><td>MAE</td><td>0.034</td><td>JARVIS</td><td>alignn_ff_db</td><td>307111</td></tr><tr><td><a href="./AI" target="_blank">AI</a></td><td><a href="./AI/ImageClass" target="_blank">ImageClass</a></td><td><a href="./AI/ImageClass/bravais_class" target="_blank">bravais_class</a></td><td><a href="https://arxiv.org/abs/2212.02586" target="_blank">densenet_model</a></td><td>ACC</td><td>0.83</td><td>JARVIS</td><td>stem_2d_image</td><td>9150</td></tr><tr><td><a href="./AI" target="_blank">AI</a></td><td><a href="./AI/TextClass" target="_blank">TextClass</a></td><td><a href="./AI/TextClass/categories" target="_blank">categories</a></td><td><a href="https://arxiv.org/abs/2209.08203" target="_blank">logisticreg_model</a></td><td>ACC</td><td>0.86</td><td>JARVIS</td><td>arXiv</td><td>100994</td></tr><tr><td><a href="./ES" target="_blank">ES</a></td><td><a href="./ES/SinglePropertyPrediction" target="_blank">SinglePropertyPrediction</a></td><td><a href="./ES/SinglePropertyPrediction/bandgap" target="_blank">bandgap</a></td><td><a href="https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.9b02166" target="_blank">vasp_tbmbj</a></td><td>MAE</td><td>0.498</td><td>JARVIS</td><td>dft_3d</td><td>54</td></tr><tr><td><a href="./ES" target="_blank">ES</a></td><td><a href="./ES/SinglePropertyPrediction" target="_blank">SinglePropertyPrediction</a></td><td><a href="./ES/SinglePropertyPrediction/Tc_supercon" target="_blank">Tc_supercon</a></td><td><a href="https://www.nature.com/articles/s41524-022-00933-1" target="_blank">qe_pbesol_gbrv</a></td><td>MAE</td><td>3.378</td><td>JARVIS</td><td>dft_3d</td><td>14</td></tr><tr><td><a href="./ES" target="_blank">ES</a></td><td><a href="./ES/Spectra" target="_blank">Spectra</a></td><td><a href="./ES/Spectra/dielectric_function" target="_blank">dielectric_function</a></td><td><a href="https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.9b02166" target="_blank">vasp_tbmbj</a></td><td>MULTIMAE</td><td>11.52</td><td>JARVIS</td><td>dft_3d</td><td>4</td></tr><tr><td><a href="./QC" target="_blank">QC</a></td><td><a href="./QC/EigenSolver" target="_blank">EigenSolver</a></td><td><a href="./QC/EigenSolver/electron_bands" target="_blank">electron_bands</a></td><td><a href="https://iopscience.iop.org/article/10.1088/1361-648X/ac1154/meta" target="_blank">qiskit_vqd_SU2</a></td><td>MULTIMAE</td><td>0.003</td><td>JARVIS</td><td>dft_3d</td><td>1</td></tr><!--table_content--></table>

<a name="add"></a>
# Adding benchmarks and datasets

To get started, first fork this repository by clicking on the [`Fork`](https://github.com/knc6/jarvis_leaderboard/fork) button. 

Then, clone your forked repository and install the project. Note instead of knc6, use your own username,

```
git clone https://github.com/knc6/jarvis_leaderboard
cd jarvis_leaderboard
python setup.py develop
```

## A) Adding model benchmarks to existing dataset

     To add a new benchmark, 

     1) Populate the dataset for a particular exisiting benchmar e.g.:
     `python jarvis_leaderboard/populate_data.py --benchmark_file SinglePropertyPrediction-test-exfoliation_energy-dft_3d-AI-mae --output_path=Out`
      This will generate an `id_prop.csv` file in the `Out` directory and other pertinent files such as POSCAR files for atomistic properties.
      The code will also print number of training, val and test samples.
      For methods other than AI method, only test set is provided.
      The reference data for ES is from experiments only.

     2) Develop your model(s) using this dataset, e.g.:
     `pip install alignn`
     `train_folder.py --root_dir "Out" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp`

     3) Create a folder in the `jarvis_leaderboard/benchmarks` folder under respective submodule e.g. `xyz_model`. 

     4) In the `xyz_model` folder, add comma-separated zip file (`.csv.zip`) file(s) corresponding to benchmark(s), 
     e.g. `SinglePropertyPrediction-test-exfoliation_energy-dft_2d-AI-mae.csv.zip` for `exfoliation_energy` in `dft_3d`
     dataset for `test` split using an `AI` (artificial intelligence method) with 
     `mae` (mean absolute error) metric for `SinglePropertyPrediction` (single property prediction) task. 
     Therefore, the filename should have these six components. 

     Note the word: `SinglePropertyPrediction`: task type, `test`, property: `exfoliation_energy`, dataset: `dft_3d`, 
     method: `AI`, metric: `mae` have been joined with '-' sign. 
     This format should be used for consistency in webpage generation.
     The test data splits are pre-determined, if the exact test IDs are not used, then the code might result in errors. 


     5) Add at least two columns: `id` and `prediction` in the csv file using your model. 
     The `jarvis_leaderboard/rebuild.py` script will parse the data in the csv.zip file, and
     will calculate and analyze several metrics. 
     The `id `should be identifier in the test split set and `prediction` is your model prediction.

     e.g.: for the above alignn example:
     `cp temp/prediction_results_test_set.csv SinglePropertyPrediction-test-exfoliation_energy-dft_2d-AI-mae.csv`
     Then zip the file:
     `zip SinglePropertyPrediction-test-exfoliation_energy-dft_2d-AI-mae.csv.zip SinglePropertyPrediction-test-exfoliation_energy-dft_2d-AI-mae.csv`

     We recommend to name this folder as your model name, e.g. `alignn_models`, `cfid_models`, `cgcnn_models` etc. 

     6) Add metadata info in the `metadata.json` file, an example is given in the 
     `jarvis_leaderboard/benchmarks/alignn_models/metadata.json`. 
     Also, add a `run.py` and `run.sh` scripts to reproduce the model predictions.
   
     The `project_url` in metadata.json should have link to a paper/GitHub URL.

     7) Make a pull-request to the original repo.

## B) Adding model benchmarks and a new dataset

     To add a new dataset

     1) Create a `json.zip` file in the `jarvis_leaderboard/dataset` folder under respective submodule 
     e.g. `jarvis_leaderboard/dataset/AI/SinglePropertyPrediction/dft_3d_exfoliation_energy.json.zip`.

     2) In the `.json` file should have `train`, `val`, `test` keys with array of ids and their values. 
     An example for creating such a file is provided in `jarvis_leaderboard/dataset/AI/SinglePropertyPrediction/format_data.py` 

     3) Add a `.md` file in `docs` folder with path to respective submodule e.g., 
     `docs/AI/SinglePropertyPrediction/exfoliation_energy.md` 

# License
   This template is served under the NIST license.  
   Read the [LICENSE] file for more info.