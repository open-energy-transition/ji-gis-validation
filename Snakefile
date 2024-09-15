# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
sys.path.append("./scripts")
from _helpers import RESULTS_DIR


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


rule validate_all:
    input:
        expand(RESULTS_DIR + "validation/"
            + "validation_{countries}_{planning_horizon}.xlsx",
            **config["validation"],
        ),
