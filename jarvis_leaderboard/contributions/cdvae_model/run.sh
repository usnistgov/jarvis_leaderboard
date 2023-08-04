#!/bin/bash
python run.py
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=$PWD
export WABDB=$PWD/WABDB
export WABDB_DIR=$PWD/WABDB
export HYDRA_JOBS=$PWD/HYDRA_JOBS
python cdvae/run.py data=carbon expname=carbon_test02 model.predict_property=True

python scripts/evaluate.py --n_step_each 5 --num_batches_to_samples 5 --batch_size 5 --model_path "/wrk/knc6/Software/cdvae_pip/cdvae/HYDRA_JOBS/singlerun/2023-07-30/carbon_test02" --tasks opt gen recon


python scripts/compute_metrics.py -root_path "/wrk/knc6/Software/cdvae_pip/cdvae/HYDRA_JOBS/singlerun/2023-07-30/carbon_test02" --tasks   gen recon

#python cdvae/run.py data=supercon expname=supercon_test02 model.predict_property=True
#python cdvae/run.py data=perov expname=perov_test02 model.predict_property=True

