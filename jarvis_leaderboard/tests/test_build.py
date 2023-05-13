from jarvis_leaderboard.rebuild import rebuild_pages
from jarvis_leaderboard.rebuild import get_metric_value,get_results

def test_check_errors():
    errors=rebuild_pages()
    assert len(errors)==0
def test_alignn_exfo():
    names,vals=get_results(bench_name='AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip')
    assert len(names)>6      
