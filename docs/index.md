[LICENSE]: https://github.com/usnistgov/jarvis/blob/master/LICENSE.rst
<!--
<div class="menu-logo">
    <img src="https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/images/nist-logo.jpg" alt="" width="100" height="auto"/>
</div>
-->

<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}

* {
  box-sizing: border-box;
}

body {
  font-family: Arial, Helvetica, sans-serif;
}

/* Float four columns side by side */
.column {
  float: left;
  width: 25%;
  padding: 0 10px;
}

/* Remove extra left and right margins, due to padding */
.row {margin: 0 -5px;}

/* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}

/* Responsive columns */
@media screen and (max-width: 600px) {
  .column {
    width: 100%;
    display: block;
    margin-bottom: 20px;
  }
}

/* Style the counter cards */
.card {
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
  padding: 20px;
  text-align: center;
  background-color: #f1f1f1;
  margin-bottom: 20px;
}

</style>


<h1 style="text-align:center;">Explore State-of-the-Art Materials Design Methods</h1>
<div class="row">
  <div class="column">
    <div class="card">
<h3>Artificial intelligence (AI)</h3><p>Contributions: 745</p>

      <a href="./AI" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Electronic Struct. (ES)</h3><p>Contributions: 742</p>
      <a href="./ES" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Force-field (FF)/potentials</h3><p>Contributions 282</p>
      <a href="./FF" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Quantum Comput. (QC) </h3><p>Contributions: 6</p>
      <a href="./QC" >See All Benchmarks</a>
    </div>
  </div>

</div>
<div class="row">

  <div class="column">
    <div class="card">
<h3>Experiments (EXP)</h3><p>Contributions: 25</p>
      <a href="./EXP" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <h3>Example Notebooks</h3><p>Notebooks: 16</p>
      <a href="./notebooks" >See All Notebooks</a>
    </div>
  </div>


  <div class="column">
    <div class="card">
<h3>Methodologies</h3><p>Available Methods:411</p>
      <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions">Learn more</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Contribution Guide</h3><p>Contributors: <a href="https://github.com/usnistgov/jarvis_leaderboard/graphs/contributors" >26</a></p>
      <a href="./guide/guide_short" >Learn more</a>
    </div>
  </div>

</div>

---

!!! Reference

    [JARVIS-Leaderboard: a large scale benchmark of materials design methods, Nature npj Computational Materials volume 10, 93 (2024)](https://www.nature.com/articles/s41524-024-01259-w)



<div class="row">
<div class="column">
<h1 id="table-of-contents">Table of Contents</h1>
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#example">Example benchmarks</a></li>
<li><a href="#license">License</a></li>
<li><a href="#help">Help</a></li>
</ul> 
</div>
<!--
<div class="column">
<iframe width="420" align="center" height="215" src="https://www.youtube.com/embed/QDx3jSIwpMo?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>
</div>
-->
</div>
<a name="intro"></a>
# Introduction

This project provides benchmark performances of various methods for materials science applications using the datasets available in [JARVIS-Tools databases](https://pages.nist.gov/jarvis/databases/). Some of the categories are: [Artificial Intelligence (AI)](./AI), [Electronic Structure (ES)](./ES), [Force-field (FF)](./FF), [Quantum Computation (QC)](./QC) and [Experiments (EXP)](./EXP). A variety of properties are included in the benchmark.

Typically, codes are kept in platforms like GitHub/GitLab, and data is stored in repositories like Zenodo/Figshare/NIST Materials Data. We recommend keeping the benchmarks in the JARVIS-Leaderboard to enhance reproducibility and transparency.
In addition to prediction results, we aim to capture the underlying software, hardware, and instrumental frameworks to enhance reproducibility. This project is a part of the [NIST-JARVIS](https://jarvis.nist.gov) infrastructure.
As a minimum check, we test rebuilding of the leaderboard and installations of software using GitHub actions.

![Leaderboard actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/test_build.yml/badge.svg)
![Leaderboard AI installation actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/install_ai.yml/badge.svg)
![Leaderboard ES installation actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/install_es.yml/badge.svg)
![Leaderboard FF installation actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/install_ff.yml/badge.svg)
![Leaderboard QC installation actions](https://github.com/usnistgov/jarvis_leaderboard/actions/workflows/install_qc.yml/badge.svg)


<!--number_of_benchmarks--> - Number of benchmarks: 308

<!--number_of_contributions--> - Number of contributions: 1799

<!--number_of_datapoints--> - Number of datapoints: 8748771

<!-- [Learn how to add benchmarks below](#add) -->
<!-- <p style="text-align:center;"><img align="middle" src="https://www.ctcms.nist.gov/~knc6/images/logo/jarvis-mission.png"  width="40%" height="20%"></p>-->




A brief summary table is given below:


<!--summary_table--><table style="width:100%" id="j_table"><thead><td>Category/Sub-cat.</td><td>SinglePropertyPrediction</td><td>SinglePropertyClass</td><td>MLFF</td><td>TextClass</td><td>TokenClass</td><td>TextSummary</td><td>TextGen</td><td>AtomGen</td><td>ImageClass</td><td>Spectra</td><td>EigenSolver</td><tr><td>AI</td><td><a href="./AI/SinglePropertyPrediction" target="_blank">569</a></td><td><a href="./AI/SinglePropertyClass" target="_blank">21</a></td><td><a href="./AI/MLFF" target="_blank">116</a></td><td><a href="./AI/TextClass" target="_blank">28</a></td><td><a href="./AI/TokenClass" target="_blank">1</a></td><td><a href="./AI/TextSummary" target="_blank">1</a></td><td><a href="./AI/TextGen" target="_blank">3</a></td><td><a href="./AI/AtomGen" target="_blank">3</a></td><td><a href="./AI/ImageClass" target="_blank">2</a></td><td><a href="./AI/Spectra" target="_blank">1</a></td><td>-</td><tr><tr><td>ES</td><td><a href="./ES/SinglePropertyPrediction" target="_blank">732</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./ES/Spectra" target="_blank">10</a></td><td>-</td><tr><tr><td>FF</td><td><a href="./FF/SinglePropertyPrediction" target="_blank">282</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><tr><tr><td>QC</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./QC/EigenSolver" target="_blank">6</a></td><tr><tr><td>EXP</td><td><a href="./EXP/SinglePropertyPrediction" target="_blank">7</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./EXP/Spectra" target="_blank">18</a></td><td>-</td><tr></table>





<a name="example"></a>
# Example benchmarks
Click on the entries in the Benchmark column. You'll be able to see performance comparison, methods available for each benchmark, CSV file submitted for the contribution, JSON file for ground trutch data, run.sh script for running the method and Info for metadata associated with the method.

<!--table_content--><table style="width:100%" id="j_table"><thead><tr><th>Category</th><th>Sub-category</th><th>Benchmark</th><th>Method</th><th>Metric</th><th>Score</th><th>Team</th><th>Dataset</th><th>Size</th></tr></thead><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/ssub_formula_energy" target="_blank">ssub_formula_energy</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/ElemNet2_TL" target="_blank">ElemNet2_TL</a></td><td>MAE</td><td>0.0924</td><td>NorthWestern_University</td><td>ssub</td><td>1726</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/dft_3d_formation_energy_peratom" target="_blank">dft_3d_formation_energy_peratom</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/kgcnn_coGN" target="_blank">kgcnn_coGN</a></td><td>MAE</td><td>0.0271</td><td>kgcnn</td><td>dft_3d</td><td>55713</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/dft_3d_optb88vdw_bandgap" target="_blank">dft_3d_optb88vdw_bandgap</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/kgcnn_coGN" target="_blank">kgcnn_coGN</a></td><td>MAE</td><td>0.1219</td><td>kgcnn</td><td>dft_3d</td><td>55713</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/dft_3d_optb88vdw_total_energy" target="_blank">dft_3d_optb88vdw_total_energy</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/kgcnn_coGN" target="_blank">kgcnn_coGN</a></td><td>MAE</td><td>0.0262</td><td>kgcnn</td><td>dft_3d</td><td>55713</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/dft_3d_bulk_modulus_kv" target="_blank">dft_3d_bulk_modulus_kv</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/kgcnn_coNGN" target="_blank">kgcnn_coNGN</a></td><td>MAE</td><td>8.7022</td><td>kgcnn</td><td>dft_3d</td><td>19680</td></tr><tr><td>AI</td><td>SinglePropertyClass</td><td><a href="./AI/SinglePropertyClass/dft_3d_optb88vdw_bandgap" target="_blank">dft_3d_optb88vdw_bandgap</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost" target="_blank">matminer_xgboost</a></td><td>ACC</td><td>0.9361</td><td>UofT</td><td>dft_3d</td><td>55713</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/qm9_std_jctc_LUMO" target="_blank">qm9_std_jctc_LUMO</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_model" target="_blank">alignn_model</a></td><td>MAE</td><td>0.0175</td><td>ALIGNN</td><td>qm9_std_jctc</td><td>130829</td></tr><tr><td>AI</td><td>SinglePropertyPrediction</td><td><a href="./AI/SinglePropertyPrediction/hmof_max_co2_adsp" target="_blank">hmof_max_co2_adsp</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost" target="_blank">matminer_xgboost</a></td><td>MAE</td><td>0.4622</td><td>UofT</td><td>hmof</td><td>137651</td></tr><tr><td>AI</td><td>MLFF</td><td><a href="./AI/MLFF/alignn_ff_db_energy" target="_blank">alignn_ff_db_energy</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignnff_pretrained_wt0.1" target="_blank">alignnff_pretrained_wt0.1</a></td><td>MAE</td><td>0.0342</td><td>JARVIS</td><td>alignn_ff_db</td><td>307111</td></tr><tr><td>AI</td><td>MLFF</td><td><a href="./AI/MLFF/mlearn_Si_forces" target="_blank">mlearn_Si_forces</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignnff_mlearn_wt1" target="_blank">alignnff_mlearn_wt1</a></td><td>MULTIMAE</td><td>0.06942387617720659</td><td>JARVIS</td><td>mlearn_Si</td><td>239</td></tr><tr><td>AI</td><td>ImageClass</td><td><a href="./AI/ImageClass/stem_2d_image_bravais_class" target="_blank">stem_2d_image_bravais_class</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/densenet_model" target="_blank">densenet_model</a></td><td>ACC</td><td>0.8304</td><td>JARVIS</td><td>stem_2d_image</td><td>9150</td></tr><tr><td>AI</td><td>TextClass</td><td><a href="./AI/TextClass/arXiv_categories" target="_blank">arXiv_categories</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/svc_model_text_title_abstract" target="_blank">svc_model_text_title_abstract</a></td><td>ACC</td><td>0.9082</td><td>ChemNLP</td><td>arXiv</td><td>100994</td></tr><tr><td>FF</td><td>SinglePropertyPrediction</td><td><a href="./FF/SinglePropertyPrediction/dft_3d_bulk_modulus_JVASP_816_Al" target="_blank">dft_3d_bulk_modulus_JVASP_816_Al</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/2017--Kim-J-S--Pt-Al--LAMMPS--ipr1" target="_blank">2017--Kim-J-S--Pt-Al--LAMMPS--ipr1</a></td><td>MAE</td><td>0.0114</td><td>IPR</td><td>dft_3d</td><td>1</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_bulk_modulus_JVASP_816_Al" target="_blank">dft_3d_bulk_modulus_JVASP_816_Al</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_optcx13" target="_blank">vasp_optcx13</a></td><td>MAE</td><td>0.2</td><td>JARVIS</td><td>dft_3d</td><td>1</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_bulk_modulus" target="_blank">dft_3d_bulk_modulus</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_opt86b" target="_blank">vasp_opt86b</a></td><td>MAE</td><td>4.6619</td><td>JARVIS</td><td>dft_3d</td><td>21</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_bulk_modulus_JVASP_1002_Si" target="_blank">dft_3d_bulk_modulus_JVASP_1002_Si</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_scan" target="_blank">vasp_scan</a></td><td>MAE</td><td>0.669</td><td>JARVIS</td><td>dft_3d</td><td>1</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_bandgap" target="_blank">dft_3d_bandgap</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_tbmbj" target="_blank">vasp_tbmbj</a></td><td>MAE</td><td>0.4981</td><td>JARVIS</td><td>dft_3d</td><td>54</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_bandgap_JVASP_1002_Si" target="_blank">dft_3d_bandgap_JVASP_1002_Si</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/gpaw_gllbsc" target="_blank">gpaw_gllbsc</a></td><td>MAE</td><td>0.0048</td><td>GPAW</td><td>dft_3d</td><td>1</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_epsx" target="_blank">dft_3d_epsx</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_optb88vdw_linopt" target="_blank">vasp_optb88vdw_linopt</a></td><td>MAE</td><td>1.4638</td><td>JARVIS</td><td>dft_3d</td><td>16</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_Tc_supercon_JVASP_1151_MgB2" target="_blank">dft_3d_Tc_supercon_JVASP_1151_MgB2</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/qe_pbesol_gbrv" target="_blank">qe_pbesol_gbrv</a></td><td>MAE</td><td>6.3148</td><td>JARVIS</td><td>dft_3d</td><td>1</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_Tc_supercon" target="_blank">dft_3d_Tc_supercon</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/qe_pbesol_gbrv" target="_blank">qe_pbesol_gbrv</a></td><td>MAE</td><td>3.3785</td><td>JARVIS</td><td>dft_3d</td><td>14</td></tr><tr><td>ES</td><td>SinglePropertyPrediction</td><td><a href="./ES/SinglePropertyPrediction/dft_3d_slme" target="_blank">dft_3d_slme</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_tbmbj" target="_blank">vasp_tbmbj</a></td><td>MAE</td><td>5.0925</td><td>JARVIS</td><td>dft_3d</td><td>5</td></tr><tr><td>ES</td><td>Spectra</td><td><a href="./ES/Spectra/dft_3d_dielectric_function" target="_blank">dft_3d_dielectric_function</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/vasp_tbmbj" target="_blank">vasp_tbmbj</a></td><td>MULTIMAE</td><td>2.8799621766740207</td><td>JARVIS</td><td>dft_3d</td><td>4</td></tr><tr><td>QC</td><td>EigenSolver</td><td><a href="./QC/EigenSolver/dft_3d_electron_bands_JVASP_816_Al_WTBH" target="_blank">dft_3d_electron_bands_JVASP_816_Al_WTBH</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/qiskit_vqd_SU2_c6" target="_blank">qiskit_vqd_SU2_c6</a></td><td>MULTIMAE</td><td>0.002963733593749998</td><td>JARVIS</td><td>dft_3d</td><td>1</td></tr><tr><td>EXP</td><td>Spectra</td><td><a href="./EXP/Spectra/dft_3d_XRD_JVASP_19821_MgB2" target="_blank">dft_3d_XRD_JVASP_19821_MgB2</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/bruker_d8" target="_blank">bruker_d8</a></td><td>MULTIMAE</td><td>0.020040003548149166</td><td>MML-BrukerD8</td><td>dft_3d</td><td>1</td></tr><tr><td>EXP</td><td>Spectra</td><td><a href="./EXP/Spectra/nist_isodb_co2_RM_8852" target="_blank">nist_isodb_co2_RM_8852</a></td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/10.1007s10450-018-9958-x.Lab01" target="_blank">10.1007s10450-018-9958-x.Lab01</a></td><td>MULTIMAE</td><td>0.02129168060976213</td><td>FACTlab</td><td>nist_isodb</td><td>1</td></tr><!--table_content--></table>









       
<a name="help"></a>
# Help

   If you have a question/suggestion, raise a [GitHub issue](https://github.com/usnistgov/jarvis_leaderboard/issues) or submit a [Google form](https://forms.gle/giDEnfNmkaU5BhBw9) request.


<a name="license"></a>
# License

   This template is served under the NIST license.  
   Read the [LICENSE] file for more info.