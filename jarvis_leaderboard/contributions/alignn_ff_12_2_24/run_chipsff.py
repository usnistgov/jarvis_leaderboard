#!/usr/bin/env python
import os
import pandas as pd
import plotly.express as px
import argparse
from jarvis.db.jsonutils import loadjson
from chipsff.config import CHIPSFFConfig
from tqdm import tqdm
from chipsff.general_material_analyzer import MaterialsAnalyzer
from chipsff.alignn_ff_db_analyzer import AlignnFFForcesAnalyzer
from chipsff.mlearn_db_analyzer import MLearnForcesAnalyzer
from chipsff.mptraj_analyzer import MPTrjAnalyzer
from chipsff.scaling_analyzer import ScalingAnalyzer


def analyze_multiple_structures(
    jid_list, calculator_types, chemical_potentials_file, **kwargs
):
    """
    Analyzes multiple structures (by JID) with multiple calculators
    and aggregates error metrics.
    """
    composite_error_data = {}

    for calculator_type in calculator_types:
        # List to store individual error DataFrames
        error_dfs = []

        for jid in tqdm(jid_list, total=len(jid_list)):
            print(f"Analyzing {jid} with {calculator_type}...")
            # Fetch calculator-specific settings
            calc_settings = kwargs.get("calculator_settings", {}).get(
                calculator_type, {}
            )
            analyzer = MaterialsAnalyzer(
                jid=jid,
                calculator_type=calculator_type,
                chemical_potentials_file=chemical_potentials_file,
                bulk_relaxation_settings=kwargs.get("bulk_relaxation_settings"),
                phonon_settings=kwargs.get("phonon_settings"),
                properties_to_calculate=kwargs.get("properties_to_calculate"),
                use_conventional_cell=kwargs.get("use_conventional_cell", False),
                surface_settings=kwargs.get("surface_settings"),
                defect_settings=kwargs.get("defect_settings"),
                phonon3_settings=kwargs.get("phonon3_settings"),
                md_settings=kwargs.get("md_settings"),
                calculator_settings=calc_settings,  # Pass calculator-specific settings
            )
            # Run analysis and get error data
            error_dat = analyzer.run_all()
            error_df = pd.DataFrame([error_dat])
            error_dfs.append(error_df)

        # Concatenate all error DataFrames
        all_errors_df = pd.concat(error_dfs, ignore_index=True)

        # Compute composite errors by ignoring NaN values
        composite_error = all_errors_df.mean(skipna=True).to_dict()

        # Store the composite error data for this calculator type
        composite_error_data[calculator_type] = composite_error

    # Once all materials and calculators have been processed, create a DataFrame
    composite_df = pd.DataFrame(composite_error_data).transpose()

    # Plot the composite scorecard
    plot_composite_scorecard(composite_df)

    # Save the composite dataframe
    composite_df.to_csv("composite_error_data.csv", index=True)


def analyze_multiple_interfaces(
    film_jid_list,
    substrate_jid_list,
    calculator_types,
    chemical_potentials_file,
    film_index="1_1_0",
    substrate_index="1_1_0",
):
    """
    Analyzes multiple film/substrate JIDs for interface properties
    with multiple calculators.
    """
    for calculator_type in calculator_types:
        for film_jid in film_jid_list:
            for substrate_jid in substrate_jid_list:
                print(
                    f"Analyzing interface between {film_jid} and {substrate_jid} "
                    f"with {calculator_type}..."
                )
                analyzer = MaterialsAnalyzer(
                    calculator_type=calculator_type,
                    chemical_potentials_file=chemical_potentials_file,
                    film_jid=film_jid,
                    substrate_jid=substrate_jid,
                    film_index=film_index,
                    substrate_index=substrate_index,
                )
                analyzer.analyze_interfaces()


def plot_composite_scorecard(df):
    """
    Creates a heatmap scorecard of error metrics for multiple calculator types.
    """
    fig = px.imshow(
        df, text_auto=True, aspect="auto", labels=dict(color="Error")
    )

    # Update layout for larger font sizes
    fig.update_layout(
        font=dict(size=24),
        coloraxis_colorbar=dict(title_font=dict(size=18), tickfont=dict(size=18)),
    )
    # Optionally adjust the text font size in cells
    fig.update_traces(textfont=dict(size=18))

    fname_plot = "composite_error_scorecard.png"
    fig.write_image(fname_plot)
    fig.show()


def main():
    import pprint

    parser = argparse.ArgumentParser(description="Run Materials Analyzer")
    parser.add_argument(
        "--input_file",
        default="input.json",
        type=str,
        help="Path to the input configuration JSON file",
    )
    args = parser.parse_args()

    # Load the JSON file, parse into CHIPSFFConfig
    input_file = loadjson(args.input_file)
    input_file_data = CHIPSFFConfig(**input_file)
    pprint.pprint(input_file_data.dict())

    # 1) Check if scaling test is requested
    if input_file_data.scaling_test:
        print("Running scaling test...")
        scaling_analyzer = ScalingAnalyzer(input_file_data)
        scaling_analyzer.run()
        return  # Done.

    # 2) If a local structure_path is provided, handle that scenario
    if input_file_data.structure_path:
        # Decide which calculators to run
        if input_file_data.calculator_type:
            calc_list = [input_file_data.calculator_type]
        elif input_file_data.calculator_types:
            calc_list = input_file_data.calculator_types
        else:
            raise ValueError("No calculator_type or calculator_types specified for structure_path.")

        for calc_type in calc_list:
            print(f"Analyzing local file {input_file_data.structure_path} with {calc_type}...")
            calc_settings = input_file_data.calculator_settings.get(calc_type, {})
            analyzer = MaterialsAnalyzer(
                structure_path=input_file_data.structure_path,
                calculator_type=calc_type,
                chemical_potentials_file=input_file_data.chemical_potentials_file,
                bulk_relaxation_settings=input_file_data.bulk_relaxation_settings,
                phonon_settings=input_file_data.phonon_settings,
                properties_to_calculate=input_file_data.properties_to_calculate,
                use_conventional_cell=input_file_data.use_conventional_cell,
                surface_settings=input_file_data.surface_settings,
                defect_settings=input_file_data.defect_settings,
                phonon3_settings=input_file_data.phonon3_settings,
                md_settings=input_file_data.md_settings,
                calculator_settings=calc_settings,
            )
            analyzer.run_all()

        # Return after finishing local-file scenario to avoid
        # mixing with JID-based or interface-based logic below
        return

    # 3) Otherwise, proceed with JID-based or interface-based logic

    # Determine the list of JIDs
    if input_file_data.jid:
        jid_list = [input_file_data.jid]
    elif input_file_data.jid_list:
        jid_list = input_file_data.jid_list
    else:
        jid_list = []

    # Determine the list of calculators
    if input_file_data.calculator_type:
        calculator_list = [input_file_data.calculator_type]
    elif input_file_data.calculator_types:
        calculator_list = input_file_data.calculator_types
    else:
        calculator_list = []

    # Handle film and substrate IDs for interface analysis
    film_jids = input_file_data.film_jid if input_file_data.film_jid else []
    substrate_jids = (
        input_file_data.substrate_jid
        if input_file_data.substrate_jid
        else []
    )

    # Scenario 5: Batch Processing for Multiple JIDs and Calculators
    if input_file_data.jid_list and input_file_data.calculator_types:
        analyze_multiple_structures(
            jid_list=input_file_data.jid_list,
            calculator_types=input_file_data.calculator_types,
            chemical_potentials_file=input_file_data.chemical_potentials_file,
            bulk_relaxation_settings=input_file_data.bulk_relaxation_settings,
            phonon_settings=input_file_data.phonon_settings,
            properties_to_calculate=input_file_data.properties_to_calculate,
            use_conventional_cell=input_file_data.use_conventional_cell,
            surface_settings=input_file_data.surface_settings,
            defect_settings=input_file_data.defect_settings,
            phonon3_settings=input_file_data.phonon3_settings,
            md_settings=input_file_data.md_settings,
            calculator_settings=input_file_data.calculator_settings,
        )

    else:
        # Scenario 1 & 3: Single or Multiple JIDs with Single or Multiple Calculators
        if jid_list and tqdm(calculator_list, total=len(calculator_list)):
            for jid in tqdm(jid_list, total=len(jid_list)):
                for calculator_type in calculator_list:
                    print(f"Analyzing {jid} with {calculator_type}...")
                    # Fetch calculator-specific settings
                    calc_settings = (
                        input_file_data.calculator_settings.get(calculator_type, {})
                    )
                    analyzer = MaterialsAnalyzer(
                        jid=jid,
                        calculator_type=calculator_type,
                        chemical_potentials_file=input_file_data.chemical_potentials_file,
                        bulk_relaxation_settings=input_file_data.bulk_relaxation_settings,
                        phonon_settings=input_file_data.phonon_settings,
                        properties_to_calculate=input_file_data.properties_to_calculate,
                        use_conventional_cell=input_file_data.use_conventional_cell,
                        surface_settings=input_file_data.surface_settings,
                        defect_settings=input_file_data.defect_settings,
                        phonon3_settings=input_file_data.phonon3_settings,
                        md_settings=input_file_data.md_settings,
                        calculator_settings=calc_settings,
                    )
                    analyzer.run_all()

    # Scenario 2 & 4: Interface Calculations (Multiple Calculators and/or JIDs)
    if film_jids and substrate_jids and calculator_list:
        for film_jid, substrate_jid in zip(film_jids, substrate_jids):
            for calculator_type in calculator_list:
                print(
                    f"Analyzing interface between {film_jid} and {substrate_jid} "
                    f"with {calculator_type}..."
                )
                # Fetch calculator-specific settings
                calc_settings = input_file_data.calculator_settings.get(calculator_type, {})
                analyzer = MaterialsAnalyzer(
                    calculator_type=calculator_type,
                    chemical_potentials_file=input_file_data.chemical_potentials_file,
                    film_jid=film_jid,
                    substrate_jid=substrate_jid,
                    film_index=input_file_data.film_index,
                    substrate_index=input_file_data.substrate_index,
                    bulk_relaxation_settings=input_file_data.bulk_relaxation_settings,
                    phonon_settings=input_file_data.phonon_settings,
                    properties_to_calculate=input_file_data.properties_to_calculate,
                    calculator_settings=calc_settings,
                )
                analyzer.analyze_interfaces()

    # Scenario 6: MLearn Forces Comparison
    if input_file_data.mlearn_elements and input_file_data.calculator_type:
        print(
            f"Running mlearn forces comparison for elements "
            f"{input_file_data.mlearn_elements} with {input_file_data.calculator_type}..."
        )
        mlearn_analyzer = MLearnForcesAnalyzer(
            calculator_type=input_file_data.calculator_type,
            mlearn_elements=input_file_data.mlearn_elements,
            calculator_settings=input_file_data.calculator_settings.get(
                input_file_data.calculator_type, {}
            ),
        )
        mlearn_analyzer.run()

    # Scenario 7: AlignnFF Forces Comparison
    if input_file_data.alignn_ff_db and input_file_data.calculator_type:
        print(
            f"Running AlignnFF forces comparison with {input_file_data.calculator_type}..."
        )
        alignn_ff_analyzer = AlignnFFForcesAnalyzer(
            calculator_type=input_file_data.calculator_type,
            num_samples=input_file_data.num_samples,
            calculator_settings=input_file_data.calculator_settings.get(
                input_file_data.calculator_type, {}
            ),
        )
        alignn_ff_analyzer.run()

    # Scenario 8: MPTrj Forces Comparison
    if input_file_data.mptrj and input_file_data.calculator_type:
        print(
            f"Running MPTrj forces comparison with {input_file_data.calculator_type}..."
        )
        mptrj_analyzer = MPTrjAnalyzer(
            calculator_type=input_file_data.calculator_type,
            num_samples=input_file_data.num_samples,
            calculator_settings=input_file_data.calculator_settings.get(
                input_file_data.calculator_type, {}
            ),
        )
        mptrj_analyzer.run()


if __name__ == "__main__":
    main()

