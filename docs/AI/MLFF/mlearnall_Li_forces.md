# Model for Li FF forces
<!--benchmark_description--> - Description: This is an AI benchmark to evaluate how accurately a machine learning force-field (MLFF) can predict the forces of Li using the relaxation trajectories (energy and forces of intermediate steps) of the mlearn dataset, calculated with the PBE density functional. The dataset contains different types of chemical formula and atomic structures. Here we use multi-mean absolute error (multi-MAE) to compare MLFFs with respect to DFT (PBE) accuracy. External links: https://github.com/materialsvirtuallab/mlearn<br><div>            <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>     <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>        <div id="50a0d311-5c3e-4c7c-8438-343915c1f40d" class="plotly-graph-div" style="height:100%; width:100%;"></div>      <script type="text/javascript">                  window.PLOTLYENV=window.PLOTLYENV || {};                  if (document.getElementById("50a0d311-5c3e-4c7c-8438-343915c1f40d")) {          Plotly.newPlot(            "50a0d311-5c3e-4c7c-8438-343915c1f40d",            [{"x":["mlearn_analysis_Li_eqV2_153M_omat","mlearn_analysis_Li_eqV2_86M_omat","mlearn_analysis_Li_eqV2_31M_omat","mlearn_analysis_Li_eqV2_86M_omat_mp_salex","mlearn_analysis_Li_eqV2_31M_omat_mp_salex","mlearn_analysis_Li_orb-v2","mlearn_analysis_Li_mace-alexandria","mlearn_analysis_Li_matgl-direct","mlearn_analysis_Li_mace","mlearn_analysis_Li_matgl","mlearn_analysis_Li_chgnet","mlearn_analysis_Li_sevennet","mlearn_analysis_Li_alignn_ff"],"y":[0.0042284903650996585,0.00426599106514172,0.004447577745790214,0.005438631629283516,0.006596184239007061,0.01813756956797843,0.025130639164981892,0.03102841476565417,0.03500126641942033,0.04245423912614389,0.044763929795324155,0.050912809549211124,0.1895976271397354],"type":"bar"}],            {"title":{"text":"AI-MLFF-forces-mlearnall_Li-test-multimae","x":0.5},"yaxis":{"title":{"text":"MULTIMAE (forces)"}},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},            {"responsive": true}          )        };              </script>    </div><br>Reference(s): [https://doi.org/10.1021/acs.jpca.9b08723](https://doi.org/10.1021/acs.jpca.9b08723), [https://github.com/usnistgov/chipsff](https://github.com/usnistgov/chipsff)<br>


<h2>Model benchmarks</h2>

<table style="width:100%" id="j_table">
 <thead>
  <tr>
<th>Model name</th><th>Dataset</th>
   <!-- <th>Method</th>-->
    <th>Multimae</th>
    <th>Team name</th>
    <th>Dataset size</th>
    <th>Date submitted</th>
    <th>Notes</th>
  </tr>
 </thead>
<!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat" target="_blank">mlearn_analysis_Li_eqV2_86M_omat</a></td><td>mlearnall_Li</td><td>0.00426599106514172</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace" target="_blank">mlearn_analysis_Li_mace</a></td><td>mlearnall_Li</td><td>0.03500126641942033</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl-direct" target="_blank">mlearn_analysis_Li_matgl-direct</a></td><td>mlearnall_Li</td><td>0.03102841476565417</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl-direct/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl-direct/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl-direct/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_alignn_ff" target="_blank">mlearn_analysis_Li_alignn_ff</a></td><td>mlearnall_Li</td><td>0.1895976271397354</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_alignn_ff/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_alignn_ff/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_alignn_ff/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat_mp_salex" target="_blank">mlearn_analysis_Li_eqV2_86M_omat_mp_salex</a></td><td>mlearnall_Li</td><td>0.005438631629283516</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat_mp_salex/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat_mp_salex/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_86M_omat_mp_salex/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat" target="_blank">mlearn_analysis_Li_eqV2_31M_omat</a></td><td>mlearnall_Li</td><td>0.004447577745790214</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_orb-v2" target="_blank">mlearn_analysis_Li_orb-v2</a></td><td>mlearnall_Li</td><td>0.01813756956797843</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_orb-v2/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_orb-v2/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_orb-v2/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl" target="_blank">mlearn_analysis_Li_matgl</a></td><td>mlearnall_Li</td><td>0.04245423912614389</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_matgl/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_sevennet" target="_blank">mlearn_analysis_Li_sevennet</a></td><td>mlearnall_Li</td><td>0.050912809549211124</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_sevennet/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_sevennet/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_sevennet/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace-alexandria" target="_blank">mlearn_analysis_Li_mace-alexandria</a></td><td>mlearnall_Li</td><td>0.025130639164981892</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace-alexandria/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace-alexandria/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_mace-alexandria/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_153M_omat" target="_blank">mlearn_analysis_Li_eqV2_153M_omat</a></td><td>mlearnall_Li</td><td>0.0042284903650996585</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_153M_omat/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_153M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_153M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_chgnet" target="_blank">mlearn_analysis_Li_chgnet</a></td><td>mlearnall_Li</td><td>0.044763929795324155</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_chgnet/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_chgnet/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_chgnet/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat_mp_salex" target="_blank">mlearn_analysis_Li_eqV2_31M_omat_mp_salex</a></td><td>mlearnall_Li</td><td>0.006596184239007061</td><td>JARVIS</td><td>270</td><td>11-22-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat_mp_salex/AI-MLFF-forces-mlearnall_Li-test-multimae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/MLFF/mlearnall_Li_forces.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat_mp_salex/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mlearn_analysis_Li_eqV2_31M_omat_mp_salex/metadata.json " target="_blank">Info</a></td></tr><!--table_content-->
</table>