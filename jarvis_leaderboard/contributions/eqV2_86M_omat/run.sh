#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs
#jid_list=('JVASP-1002')
# Define arrays of JIDs and calculators
jid_list=('JVASP-1002' 'JVASP-890' 'JVASP-39' 'JVASP-30' 'JVASP-62940' 'JVASP-20092' 'JVASP-8003' 'JVASP-1192' 'JVASP-23' 'JVASP-1195' 'JVASP-96' 'JVASP-10591' 'JVASP-1198' 'JVASP-1312' 'JVASP-133719' 'JVASP-36873' 'JVASP-1327' 'JVASP-1372' 'JVASP-1408' 'JVASP-8184' 'JVASP-1174' 'JVASP-1177' 'JVASP-1180' 'JVASP-1183' 'JVASP-1186' 'JVASP-1189' 'JVASP-91' 'JVASP-8158' 'JVASP-8118' 'JVASP-107' 'JVASP-36018' 'JVASP-36408' 'JVASP-105410' 'JVASP-1008' 'JVASP-21211' 'JVASP-1023' 'JVASP-7836' 'JVASP-9166' 'JVASP-1201' 'JVASP-85478' 'JVASP-1115' 'JVASP-1112' 'JVASP-1103' 'JVASP-1109' 'JVASP-131' 'JVASP-149916' 'JVASP-111005' 'JVASP-25' 'JVASP-1067' 'JVASP-10703' 'JVASP-104' 'JVASP-10036' 'JVASP-18983' 'JVASP-1216' 'JVASP-79522' 'JVASP-1222' 'JVASP-10037' 'JVASP-110' 'JVASP-8082' 'JVASP-1240' 'JVASP-29539' 'JVASP-54' 'JVASP-1915' 'JVASP-22694' 'JVASP-4282' 'JVASP-76195' 'JVASP-8554' 'JVASP-149871' 'JVASP-2376' 'JVASP-3510' 'JVASP-5224' 'JVASP-8559' 'JVASP-85416' 'JVASP-9117' 'JVASP-90668' 'JVASP-103127' 'JVASP-104764' 'JVASP-110231' 'JVASP-108770' 'JVASP-149906' 'JVASP-99732' 'JVASP-106686' 'JVASP-106363' 'JVASP-972' 'JVASP-825' 'JVASP-813' 'JVASP-816' 'JVASP-802' 'JVASP-1029' 'JVASP-861' 'JVASP-943' 'JVASP-963' 'JVASP-14616' 'JVASP-867' 'JVASP-14968' 'JVASP-14970' 'JVASP-19780' 'JVASP-9147' 'JVASP-34249' 'JVASP-43367' 'JVASP-113' 'JVASP-41' 'JVASP-58349' 'JVASP-34674' 'JVASP-34249' 'JVASP-32')
calculator_types=("mace")

# Loop through each JID and calculator combination
for jid in "${jid_list[@]}"; do
  for calculator in "${calculator_types[@]}"; do

    # Submit each job with a separate sbatch command, requesting a dedicated node
    sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=30-00:00:00
#SBATCH --partition=rack1,rack2e,rack3,rack4,rack4e,rack5,rack6
#SBATCH --job-name=${jid}_${calculator}
#SBATCH --output=logs/${jid}_${calculator}_%j.out
#SBATCH --error=logs/${jid}_${calculator}_%j.err

# Generate input JSON file for this combination
cat > input_${jid}_${calculator}.json <<JSON
{
  "jid": "$jid",
  "calculator_type": "$calculator",
  "chemical_potentials_file": "chemical_potentials.json",
  "properties_to_calculate": [
    "relax_structure",
    "calculate_ev_curve",
    "calculate_formation_energy",
    "calculate_elastic_tensor",
    "run_phonon_analysis",
    "analyze_surfaces",
    "analyze_defects"
  ],
  "bulk_relaxation_settings": {
    "filter_type": "ExpCellFilter",
    "relaxation_settings": {
      "fmax": 0.05,
      "steps": 200,
      "constant_volume": false
    }
  },
  "phonon_settings": {
    "dim": [2, 2, 2],
    "distance": 0.2
  },
  "use_conventional_cell": false,
  "surface_settings": {
    "indices_list": [
      [1, 0, 0],
      [1, 1, 1],
      [1, 1, 0],
      [0, 1, 1],
      [0, 0, 1],
      [0, 1, 0]
    ],
    "layers": 4,
    "vacuum": 18,
    "relaxation_settings": {
      "fmax": 0.05,
      "steps": 200,
      "constant_volume": true
    },
    "filter_type": "ExpCellFilter"
  },
  "defect_settings": {
    "generate_settings": {
      "on_conventional_cell": true,
      "enforce_c_size": 8,
      "extend": 1
    },
    "relaxation_settings": {
      "fmax": 0.05,
      "steps": 200,
      "constant_volume": true
    },
    "filter_type": "ExpCellFilter"
  },
  "phonon3_settings": {
    "dim": [2, 2, 2],
    "distance": 0.2
  },
  "md_settings": {
    "dt": 1,
    "temp0": 35,
    "nsteps0": 10,
    "temp1": 200,
    "nsteps1": 20,
    "taut": 20,
    "min_size": 10.0
  }
}
JSON

# Run the Python analysis for this JID/calculator combination
python run_chipsff.py --input_file input_${jid}_${calculator}.json

EOT

  done
done
