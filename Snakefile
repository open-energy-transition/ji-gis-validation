# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

RESULTS_DIR = "results/"


configfile: "configs/config.yaml"

wildcard_constraints:
    countries="[A-Z]{2}",
    clusters="[0-9]+",
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
        clusters=config["validation"]["clusters"],
        planning_horizon=config["validation"]["planning_horizon"],
    output:
        demand=RESULTS_DIR + "plots/" + "demand_validation_{clusters}_{countries}_{planning_horizon}.png",
        capacity=RESULTS_DIR + "plots/" + "capacity_validation_{clusters}_{countries}_{planning_horizon}.png",
        generation=RESULTS_DIR + "plots/" + "generation_validation_{clusters}_{countries}_{planning_horizon}.png",
        generation_detailed=RESULTS_DIR + "plots/" + "generation_validation_detailed_{clusters}_{countries}_{planning_horizon}.png",
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
            + "demand_validation_{clusters}_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "capacity_validation_{clusters}_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "generation_validation_{clusters}_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
        expand(RESULTS_DIR + "plots/"
            + "generation_validation_detailed_{clusters}_{countries}_{planning_horizon}.png",
            **config["validation"],
        ),
