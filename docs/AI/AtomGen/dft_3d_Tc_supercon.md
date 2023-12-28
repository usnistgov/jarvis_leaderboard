# Model for 3D superconductor design
<!--benchmark_description--> - Description: This is a benchmark to evaluate how accurately a generative model can generate atomic structures with desired super critical transition temperature (Tc) using the JARVIS-DFT (dft_3d) dataset. The dataset contains different types of chemical formula and atomic structures. Here we use root mean squared error (RMSE) for bondlengths to compare models. External links: https://github.com/txie-93/cdvae, https://arxiv.org/abs/2110.06197, https://pubs.acs.org/doi/10.1021/acs.jpclett.3c01260



<h2>Model benchmarks</h2>

<table style="width:100%" id="j_table">
 <thead>
  <tr>
<th>Model name</th><th>Dataset</th>
   <!-- <th>Method</th>-->
    <th>RMSE</th>
    <th>Team name</th>
    <th>Dataset size</th>
    <th>Date submitted</th>
    <th>Notes</th>
  </tr>
 </thead>
<!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/cdvae_model" target="_blank">cdvae_model</a></td><td>dft_3d</td><td>0.3557</td><td>CDVAE</td><td>1056</td><td>07-30-2023</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/cdvae_model/AI-AtomGen-Tc_supercon-dft_3d-test-rmse.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/AtomGen/dft_3d_Tc_supercon.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/cdvae_model/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/cdvae_model/metadata.json " target="_blank">Info</a></td></tr><!--table_content-->
</table>