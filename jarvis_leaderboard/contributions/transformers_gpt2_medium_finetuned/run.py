from chemnlp.textgen.generator import train_generator

import os
cmd='wget https://figshare.com/ndownloader/files/39768544 -O cond_mat.zip'
os.system(cmd)
train_generator(csv_file="cond_mat.zip", model_checkpoint="gpt2-medium")
