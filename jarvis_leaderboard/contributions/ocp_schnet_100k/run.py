#/wrk/knc6/oc/oc2/ocp/data/val_cgcnn_test/pred2.py
# conda activate ocp-models
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
#from ocpmodels.datasets import SinglePointLmdbDataset
import os, torch
from ase.io import read
#from ocpmodels.preprocessing import AtomsToGraphs
#from ocpmodels.models import CGCNN
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from jarvis.core.atoms import Atoms
#from ocpmodels.datasets import data_list_collater
from tqdm import tqdm
from jarvis.db.figshare import  get_request_data
import json,zipfile
import numpy as np
import pandas as pd
from jarvis.db.jsonutils import loadjson,dumpjson
import os
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
from ocpmodels.datasets import SinglePointLmdbDataset
import os, torch
from ase.io import read
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models import CGCNN,SchNet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from jarvis.core.atoms import Atoms
from ocpmodels.datasets import data_list_collater


dat = get_request_data(js_tag="ocp100k.json",url="https://figshare.com/ndownloader/files/40902845")
df = pd.DataFrame(dat)
path = "../../../../../benchmarks/AI/SinglePropertyPrediction/ocp100k_relaxed_energy.json.zip"
js_tag = "ocp100k_relaxed_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()))
val_ids = np.array(list(id_data['val'].keys()))
test_ids = np.array(list(id_data['test'].keys()))
train_df = df[df['id'].isin(train_ids)]
val_df = (df[df['id'].isin(val_ids)])#[:take_val]
test_df = (df[df['id'].isin(test_ids)])

print (test_df)
#https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/is2re/10k/base.yml
target_mean= -1.525913953781128
target_std= 2.279365062713623

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


setup(0, 1)


a2g = AtomsToGraphs(
    max_neigh=200,
    radius=6,
    r_energy=False,
    r_forces=False,
    r_distances=True,
    r_edges=True,
)
attrs = torch.load("schnet_100k.pt", map_location=device)["config"][
    "model_attributes"
]
print("attrs", attrs)
model = SchNet(
    None,
    None,
    1,
    cutoff=6.0,
    hidden_channels=attrs["hidden_channels"],
    num_interactions=attrs["num_interactions"],   
    num_gaussians=attrs["num_gaussians"],
    regress_forces=False,
    use_pbc=True,
)

model = DDP(model, device_ids=0)
model.load_state_dict(
    torch.load("schnet_100k.pt", map_location=device)["state_dict"]
)
model.to(device)
model.eval()

f=open('AI-SinglePropertyPrediction-relaxed_energy-ocp100k-test-mae.csv','w')
f.write('id,target,prediction\n')
#f.write('id,target,scaled_target,prediction\n')
print('id,actual,scaled,pred')

for ii,i in tqdm(test_df.iterrows()):
    fname=i['id']
    atoms=(Atoms.from_dict(i['atoms'])).ase_converter()
    actual=i['relaxed_energy']
    relaxed_energy = (actual-target_mean)/target_std
    scaled=relaxed_energy
    data = a2g.convert(atoms).to(device)
    batch = data_list_collater([data], otf_graph=False)
    out = model(batch)
    pred=(out[0].detach().cpu().numpy().flatten().tolist()[0])*target_std+target_mean
    line=str(fname)+','+str(actual)+','+str(pred) #+'\n'
    #line=str(i.sid)+','+str(actual)+','+str(scaled)+','+str(pred) #+'\n'
    f.write(line+'\n')
f.close()
