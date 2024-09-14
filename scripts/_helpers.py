# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from pathlib import Path
import pypsa
import logging
import yaml
import pandas as pd
import pycountry
import warnings
warnings.filterwarnings("ignore")


# get the base working directory
BASE_PATH = os.path.abspath(os.path.join(__file__ ,"../.."))
# directory to data and result folders
DATA_DIR = "data/"
PYPSA_RESULTS_DIR = "pypsa_data/results"
PYPSA_NETWORKS_DIR = "pypsa_data/networks"
RESULTS_DIR = "results/"


def create_folder(directory):
    """ Input:
            directory - path to directory
        Output:
            N/A
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Folder '{directory}' created.")
    else:
        print(f"Folder '{directory}' already exists.")


def get_solved_network_path(country_code, horizon, directory):
    """ Input:
            country_code - two-letter country code
            horizon - planning horizon
            directory - path to results directory
        Output:
            filepath - full path to network
    """
    foldername = f"{country_code}_{horizon}/networks"
    filenames = os.listdir(os.path.join(directory, foldername))
    if len(filenames) != 1:
        logging.warning(f"Only 1 network per scenario is allowed currently!")
    filepath = os.path.join(directory, foldername, filenames[0])   
    return filepath


def get_base_network_path(country_code, horizon, directory):
    """ Input:
            country_code - two-letter country code
            horizon - planning horizon
            directory - path to results directory
        Output:
            filepath - full path to base network
    """
    foldername = f"{country_code}_{horizon}/"
    filenames = os.listdir(os.path.join(directory, foldername))

    filepath = os.path.join(directory, foldername, "base.nc")
    return filepath


def load_network(filepath):
    """ Input:
            filepath - full path to the network
        Output:
            n - PyPSA network
    """
    try:
        n = pypsa.Network(filepath)
        logging.info(f"Loading {filepath}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    return n


def get_country_name(country_code):
    """ Input:
            country_code - two letter code of the country
        Output:
            country.name - corresponding name of the country
            country.alpha_3 - three letter code of the country
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name, country.alpha_3 if country else None
    except KeyError:
        return None
    

def get_country_list(directory):
    """ Input:
            directory - path to results directory
        Output:
            List of countries in two-letter format in pypsa_data/results folder
    """
    filenames = os.listdir(directory)
    country_names = set([x[:2] for x in filenames])
    return sorted(list(country_names))


def mock_snakemake(
    rulename,
    root_dir=None,
    configfiles=None,
    submodule_dir="workflow/submodules/pypsa-eur",
    **wildcards,
):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    root_dir: str/path-like
        path to the root directory of the snakemake project
    configfiles: list, str
        list of configfiles to be used to update the config
    submodule_dir: str, Path
        in case PyPSA-Eur is used as a submodule, submodule_dir is
        the path of pypsa-eur relative to the project directory.
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from pypsa.descriptors import Dict
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    script_dir = Path(__file__).parent.resolve()
    if root_dir is None:
        root_dir = script_dir.parent
    else:
        root_dir = Path(root_dir).resolve()

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if str(submodule_dir) in __file__:
        # the submodule_dir path is only need to locate the project dir
        os.chdir(Path(__file__[: __file__.find(str(submodule_dir))]))
    elif user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root"
            f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        for p in SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break
        if configfiles is None:
            configfiles = []
        elif isinstance(configfiles, str):
            configfiles = [configfiles]

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=map(Path, configfiles))
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=[])
        workflow = Workflow(
            config_settings,
            resource_settings,
            workflow_settings,
            storage_settings,
            dag_settings,
            storage_provider_settings=dict(),
        )
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake

