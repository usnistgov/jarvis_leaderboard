# Model for surface energy of CHIPSFF dataset
<!--benchmark_description--> - Description: This is a benchmark to evaluate how accurately an AI model can predict the surface energy (J/m^2) using the CHIPSFF subset of JARVIS-DFT (dft_3d) dataset. The dataset contains different types of chemical formula and atomic structures of materials common to semiconductor device components. Here we use mean absolute error (MAE) to compare models with respect to DFT accuracy.<br><div>            <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>     <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>        <div id="dc4e0b05-5903-42e5-aed9-261a506539d9" class="plotly-graph-div" style="height:100%; width:100%;"></div>      <script type="text/javascript">                  window.PLOTLYENV=window.PLOTLYENV || {};                  if (document.getElementById("dc4e0b05-5903-42e5-aed9-261a506539d9")) {          Plotly.newPlot(            "dc4e0b05-5903-42e5-aed9-261a506539d9",            [{"x":["eqV2_31M_omat_mp_salex","eqV2_86M_omat_mp_salex","orb-v2","orb-v1","eqV2_153M_omat","eqV2_86M_omat","eqV2_31M_omat","sevennet","mace","mace-alexandria","chgnet","matgl-direct","matgl","alignn_ff-5.27.2024"],"y":[0.6721,0.6796,0.7489,0.759,0.8046,0.806,0.8099,0.8946,0.9775,1.0957,1.1899,1.2636,1.3498,2.3671],"type":"bar"}],            {"title":{"text":"AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae","x":0.5},"yaxis":{"title":{"text":"MAE (surf_en)"}},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},            {"responsive": true}          )        };              </script>    </div><br>Reference(s): [https://github.com/usnistgov/chipsff](https://github.com/usnistgov/chipsff), [https://doi.org/10.48550/arXiv.2305.11842](https://doi.org/10.48550/arXiv.2305.11842), [https://www.nature.com/articles/s41524-020-00440-1](https://www.nature.com/articles/s41524-020-00440-1)<br>


<h2>Model benchmarks</h2>

<table style="width:100%" id="j_table">
 <thead>
  <tr>
    <th>Model name</th>
<th>Dataset</th>
   <!-- <th>Method</th>-->
    <th>ACC</th>
    <th>Team name</th>
    <th>Dataset size</th>
    <th>Date submitted</th>
    <th>Notes</th>
  </tr>
 </thead>
<!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/sevennet" target="_blank">sevennet</a></td><td>dft_3d_chipsff</td><td>0.8946</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/sevennet/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/sevennet/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/sevennet/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace-alexandria" target="_blank">mace-alexandria</a></td><td>dft_3d_chipsff</td><td>1.0957</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace-alexandria/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace-alexandria/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace-alexandria/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat" target="_blank">eqV2_31M_omat</a></td><td>dft_3d_chipsff</td><td>0.8099</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v2" target="_blank">orb-v2</a></td><td>dft_3d_chipsff</td><td>0.7489</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v2/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v2/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v2/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_ff-5.27.2024" target="_blank">alignn_ff-5.27.2024</a></td><td>dft_3d_chipsff</td><td>2.3671</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_ff-5.27.2024/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_ff-5.27.2024/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/alignn_ff-5.27.2024/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat_mp_salex" target="_blank">eqV2_31M_omat_mp_salex</a></td><td>dft_3d_chipsff</td><td>0.6721</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat_mp_salex/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat_mp_salex/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_31M_omat_mp_salex/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl-direct" target="_blank">matgl-direct</a></td><td>dft_3d_chipsff</td><td>1.2636</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl-direct/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl-direct/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl-direct/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl" target="_blank">matgl</a></td><td>dft_3d_chipsff</td><td>1.3498</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/matgl/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/chgnet" target="_blank">chgnet</a></td><td>dft_3d_chipsff</td><td>1.1899</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/chgnet/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/chgnet/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/chgnet/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_153M_omat" target="_blank">eqV2_153M_omat</a></td><td>dft_3d_chipsff</td><td>0.8046</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_153M_omat/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_153M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_153M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v1" target="_blank">orb-v1</a></td><td>dft_3d_chipsff</td><td>0.759</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v1/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v1/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/orb-v1/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat" target="_blank">eqV2_86M_omat</a></td><td>dft_3d_chipsff</td><td>0.806</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace" target="_blank">mace</a></td><td>dft_3d_chipsff</td><td>0.9775</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/mace/metadata.json " target="_blank">Info</a></td></tr><!--table_content--><tr><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat_mp_salex" target="_blank">eqV2_86M_omat_mp_salex</a></td><td>dft_3d_chipsff</td><td>0.6796</td><td>JARVIS</td><td>121</td><td>11-01-2024</td><td><a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat_mp_salex/AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv.zip" target="_blank">CSV</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_surf_en.json.zip" target="_blank">JSON</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat_mp_salex/run.sh " target="_blank">run.sh</a>, <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions/eqV2_86M_omat_mp_salex/metadata.json " target="_blank">Info</a></td></tr><!--table_content-->
</table>