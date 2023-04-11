from jarvis_leaderboard.rebuild import rebuild_pages

def test_check_errors():
    errors=rebuild_pages()
    assert len(errors)==0
