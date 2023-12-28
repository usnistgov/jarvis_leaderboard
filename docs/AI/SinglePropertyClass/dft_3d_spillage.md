# Model for spillage
<!--benchmark_description--> - Description: This is a benchmark to evaluate how accurately an AI model can classify a material as topological based on the DFT computed spin-orbit spillage (which is an indicator of band inversion and a signature of topological materials) from the JARVIS-DFT (dft_3d) dataset. The dataset contains different types of chemical formula and atomic structures. Here we use accuracy of classification (ACC) to compare models with respect to DFT accuracy. 


<h2>Model benchmarks</h2>

<table style="width:100%" id="j_table">
 <thead>
  <tr>
    <th>Model name</th><th>Dataset</th>
   <!-- <th>Method</th>-->
    <th>ACC</th>
    <th>Team name</th>
    <th>Dataset size</th>
    <th>Date submitted</th>
    <th>Notes</th>
  </tr>
 </thead>
<!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost" target="_blank">matminer_xgboost</a></td><td>dft_3d</td><td>0.8364</td><td>UofT</td><td>11375</td><td>05-22-2023</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost/AI-SinglePropertyClass-spillage-dft_3d-test-acc.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyClass/dft_3d_spillage.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_xgboost/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_rf" target="_blank">matminer_rf</a></td><td>dft_3d</td><td>0.8311</td><td>UofT</td><td>11375</td><td>05-22-2023</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_rf/AI-SinglePropertyClass-spillage-dft_3d-test-acc.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyClass/dft_3d_spillage.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_rf/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matminer_rf/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_model" target="_blank">alignn_model</a></td><td>dft_3d</td><td>0.8135</td><td>ALIGNN</td><td>11375</td><td>01-14-2023</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_model/AI-SinglePropertyClass-spillage-dft_3d-test-acc.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyClass/dft_3d_spillage.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_model/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_model/metadata.json " target="_blank">Info</a></td></tr><!--table_content-->
</table>