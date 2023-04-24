#! /usr/bin/env python
import os
import jarvis_leaderboard

root_dir = str(jarvis_leaderboard.__path__[0])
cwd = os.getcwd()
os.chdir(root_dir)
cmd = "mkdocs serve"
os.system(cmd)
os.chdir(cwd)
