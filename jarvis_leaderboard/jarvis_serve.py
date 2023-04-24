#! /usr/bin/env python
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
os.chdir(root_dir)
cmd = "mkdocs serve"
os.system(cmd)
os.chdir(cwd)
