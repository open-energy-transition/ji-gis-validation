# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

RESULTS_DIR = "results/"


configfile: "configs/config.yaml"

wildcard_constraints:
    countries="[A-Z]{2}",
    planning_horizon="[0-9]{4}",

localrules:
    all,


rule validate:
    params:
        countries=config["validation"]["countries"],
        planning_horizon=config["validation"]["planning_horizon"],
    output:
        table=RESULTS_DIR + "validation/" + "validation_{countries}_{planning_horizon}.xlsx",
    resources:
        mem_mb=20000,
    script:
        "scripts/validation.py"

rule plot_validation:
    params:
        countries=config["validation"]["countries"],
        planning_horizon=config["validation"]["planning_horizon"],
    output:
        demand=RESULTS_DIR + "plots/" + "demand_validation_{countries}_{planning_horizon}.png",
        capacity=RESULTS_DIR + "plots/" + "capacity_validation_{countries}_{planning_horizon}.png",
        generation=RESULTS_DIR + "plots/" + "generation_validation_{countries}_{planning_horizon}.png",
        generation_detailed=RESULTS_DIR + "plots/" + "generation_validation_detailed_{countries}_{planning_horizon}.png",
    resources:
        mem_mb=20000,
    script:
        "plots/plots_validation.py"


rule validate_all:
    input:
        expand(RESULTS_DIR + "validation/"
            + "validation_{countries}_{planning_horizon}.xlsx",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "demand_validation_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "capacity_validation_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "generation_validation_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "generation_validation_detailed_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),


rule fill_main_data:
    params:
        countries=config["database_fill"]["countries"],
        planning_horizon=config["database_fill"]["planning_horizon"],
    output:
        excel=RESULTS_DIR + "database_fill/output_{countries}_{planning_horizon}.xlsx",
    resources:
        mem_mb=8000,
    script:
        "scripts/fill_main_data.py"


rule fill_main_data_all:
    input:
        expand(RESULTS_DIR
            + "database_fill/output_{countries}_{planning_horizon}.xlsx",
            **config["database_fill"],
        ),


rule fill_investment_co2:
    output:
        excel=RESULTS_DIR + "database_fill/investment_pre_co2_reduced_{countries}.xlsx",
    resources:
        mem_mb=8000,
    script:
        "scripts/fill_investment_co2.py"


rule fill_investment_co2_all:
    input:
        expand(RESULTS_DIR
            + "database_fill/investment_pre_co2_reduced_{countries}.xlsx",
            **config["database_fill"],
        ),


rule fill_grid_data:
    params:
        countries=config["database_fill"]["countries"],
        planning_horizon=config["database_fill"]["planning_horizon"],
    output:
        scenarios=RESULTS_DIR + "database_grid_fill/scenarios.csv",
    resources:
        mem_mb=8000,
    script:
        "scripts/netcdf_power_grid_to_postgis.py"


rule fill_statistics:
    params:
        countries=config["database_fill"]["countries"],
        planning_horizon=config["validation"]["planning_horizon"],
    output:
        excel=RESULTS_DIR + "validation_results/validation_{countries}.xlsx",
    resources:
        mem_mb=8000,
    script:
        "scripts/show_statistics.py"


rule fill_statistics_all:
    input:
        expand(RESULTS_DIR + "validation_results/"
            + "validation_{countries}.xlsx",
            **config["database_fill"],
        ),
