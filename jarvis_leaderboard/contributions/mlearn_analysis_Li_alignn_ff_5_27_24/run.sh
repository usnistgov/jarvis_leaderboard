#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define arrays of JIDs, calculators, and mlearn elements
calculator_types=("alignn_ff")
mlearn_elements=("Si" "Ge" "Cu" "Ni" "Li" "Mo")

# Function to submit a job for bulk material calculations
submit_bulk_job() {
  local jid=$1
  local calculator=$2

  sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
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
    "analyze_defects",
    "run_phonon3_analysis",
    "general_melter",
    "calculate_rdf"
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
  "use_conventional_cell": true,
  "surface_settings": {
    "indices_list": [
      [0, 1, 0],
      [0, 0, 1]
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
}

# Function to submit a job for mlearn analysis
submit_mlearn_job() {
  local element=$1
  local calculator=$2

  sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --partition=rack1,rack2e,rack3,rack4,rack4e,rack5,rack6
#SBATCH --job-name=mlearn_${element}_${calculator}
#SBATCH --output=logs/mlearn_${element}_${calculator}_%j.out
#SBATCH --error=logs/mlearn_${element}_${calculator}_%j.err

# Generate input JSON file for this mlearn element/calculator combination
cat > input_mlearn_${element}_${calculator}.json <<JSON
{
  "mlearn_elements": ["$element"],
  "calculator_type": "$calculator",
  "properties_to_calculate": ["compare_mlearn_forces"]
}
JSON

# Run the Python analysis for this mlearn element/calculator combination
python run_chipsff.py --input_file input_mlearn_${element}_${calculator}.json

EOT
}

# Loop through each JID and calculator combination for bulk calculations
for jid in "${jid_list[@]}"; do
  for calculator in "${calculator_types[@]}"; do
    submit_bulk_job "$jid" "$calculator"
  done
done

# Loop through each mlearn element and calculator combination for mlearn analysis
for element in "${mlearn_elements[@]}"; do
  for calculator in "${calculator_types[@]}"; do
    submit_mlearn_job "$element" "$calculator"
  done
done


