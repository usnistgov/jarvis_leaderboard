import os

cwd = os.getcwd()
os.chdir("jarvis_leaderboard")
cmd = "mkdocs serve"
os.system(cmd)
os.chdir(cwd)
