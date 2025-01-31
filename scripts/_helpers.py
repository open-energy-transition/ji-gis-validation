# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
from pathlib import Path
import pypsa
import logging
import json
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import snakemake as sm
from pypsa.descriptors import Dict
from snakemake.script import Snakemake
import pycountry
import warnings
warnings.filterwarnings("ignore")


# Load environment variables from .env file
load_dotenv()

# get the base working directory
BASE_PATH = os.path.abspath(os.path.join(__file__, "../.."))
# directory to data and result folders
DATA_DIR = BASE_PATH + "/data/"
PYPSA_RESULTS_DIR = BASE_PATH + "/pypsa_data/results"
PYPSA_NETWORKS_DIR = BASE_PATH + "/pypsa_data/networks"
RESULTS_DIR = BASE_PATH + "/results/"
PLOTS_DIR = BASE_PATH + "/results/plots/"

PREFIX_TO_REMOVE = [
    "residential ",
    "services ",
    "urban ",
    "rural ",
    "central ",
    "decentral ",
]

RENAME_IF_CONTAINS = [
    "solid biomass CHP",
    "gas CHP",
    "gas boiler",
    "biogas",
    "solar thermal",
    "air heat pump",
    "ground heat pump",
    "resistive heater",
    "Fischer-Tropsch",
]

RENAME_IF_CONTAINS_DICT = {
    "water tanks": "TES",
    "retrofitting": "building retrofitting",
    # "H2 Electrolysis": "hydrogen storage",
    # "H2 Fuel Cell": "hydrogen storage",
    # "H2 pipeline": "hydrogen storage",
    "battery": "Battery storage",
    # "CC": "CC"
}

RENAME = {
    "Solar": "Solar PV",
    "solar": "Solar PV",
    "Sabatier": "methanation",
    "helmeth" : "methanation",
    "Offshore Wind (AC)": "Offshore wind",
    "Offshore Wind (DC)": "Offshore wind",
    "Onshore Wind": "Onshore wind",
    "offwind-ac": "Offshore wind",
    "offwind-dc": "Offshore wind",
    "Run of River": "Hydroelectricity",
    "Run of river": "Hydroelectricity",
    "Reservoir & Dam": "Hydroelectricity",
    "Pumped Hydro Storage": "Hydroelectricity",
    "PHS": "Hydroelectricity",
    "NH3": "ammonia",
    "co2 Store": "DAC",
    "co2 stored": "CO2 sequestration",
    "AC": "Transmission lines",
    "DC": "Transmission lines",
    "B2B": "Transmission lines",
    "solid biomass for industry": "solid biomass",
    "solid biomass for industry CC": "solid biomass",
    "electricity distribution grid": "distribution lines",
    "Open-Cycle Gas":"OCGT",
    "Combined-Cycle Gas":"CCGT",
    "gas": "gas storage",
    'gas pipeline new': 'gas pipeline',
    "gas for industry CC": "gas for industry",
    "SMR CC": "SMR",
    "process emissions CC": "process emissions",
    "Battery Storage": "Battery storage",
    'H2 Store': "H2 storage",
    'Hydrogen Storage': "H2 storage",
    'H2 fuel cell': "H2 storage",
    'H2 electrolysis': "H2 storage",
    'co2 sequestered': "CO2 sequestration",
    "solid biomass transport": "solid biomass",
    "uranium": "nuclear",
    "load": "Load shedding",
    "Lignite": "Coal"
}

PREFERRED_ORDER = pd.Index(
    [
        "uranium",
        "nuclear",
        "solid biomass",
        "biogas",
        "gas for industry",
        "coal for industry",
        "methanol",
        "oil",
        "lignite",
        "coal",
        "shipping oil",
        "shipping methanol",
        "naphtha for industry",
        "land transport oil",
        "kerosene for aviation",
        
        "transmission lines",
        "distribution lines",
        "gas pipeline",
        "H2 pipeline",
        
        "H2 Electrolysis",
        "H2 Fuel Cell",
        "DAC",
        "Fischer-Tropsch",
        "methanation",
        "BEV charger",
        "V2G",
        "SMR",
        "methanolisation",
        
        "battery storage",
        "gas storage",
        "H2 storage",
        "TES",
        
        "hydroelectricity",
        "OCGT",
        "CCGT",
        "onshore wind",
        "offshore wind",
        "solar PV",
        "solar thermal",
        "solar rooftop",

        "co2",
        "CO2 sequestration",
        "process emissions",

        "gas CHP",
        "solid biomass CHP",
        "resistive heater",
        "air heat pump",
        "ground heat pump",
        "gas boiler",
        "biomass boiler",
        "WWHRS",
        "building retrofitting",
        "WWHRS",
     ]
)


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


def mock_snakemake(rule_name, **wildcards):
    """
    This function is expected to be executed from the "scripts"-directory of "
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards**.

    Parameters
    ----------
    rule_name: str
        name of the rule for which the snakemake object should be generated
    wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """

    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if Path(p).exists():
            snakefile = p
            break
    workflow = sm.Workflow(
        snakefile, overwrite_configfiles=[], rerun_triggers=[]
    )  # overwrite_config=config
    workflow.include(snakefile)
    workflow.global_resources = {}
    try:
        rule = workflow.get_rule(rule_name)
    except Exception as exception:
        print(
            exception,
            f"The {rule_name} might be a conditional rule in the Snakefile.\n"
            f"Did you enable {rule_name} in the config?",
        )
        raise
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = Path(io[i]).absolute()

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
    snakemake.benchmark = job.benchmark

    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        build_directory(path)

    os.chdir(script_dir)
    return snakemake


def update_config_from_wildcards(config, w):
    if w.get("countries"):
        countries = w.countries
        config["validation"]["countries"] = countries
        config["database_fill"]["countries"] = countries
    if w.get("planning_horizon"):
        planning_horizon = w.planning_horizon
        config["validation"]["planning_horizon"] = planning_horizon
        config["database_fill"]["planning_horizon"] = planning_horizon
    return config


def build_directory(path, just_parent_directory=True):
    """
    It creates recursively the directory and its leaf directories.

    Parameters:
        path (str): The path to the file
        just_parent_directory (Boolean): given a path dir/subdir
            True: it creates just the parent directory dir
            False: it creates the full directory tree dir/subdir
    """

    # Check if the provided path points to a directory
    if just_parent_directory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_logger():
    # Configure logger to write logs to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    return logger


def connect_to_db():
    """
    Connects to the PostgreSQL database using the POST_TABLE parameter from .env file.

    Returns:
        conn: A psycopg2 connection object.
    """
    try:
        # Get POST_TABLE as a JSON string
        post_table_json = os.getenv('POST_TABLE')
        if not post_table_json:
            raise ValueError("POST_TABLE is not set in `.env` file.")

        # Parse JSON string into a Python dictionary
        db_params = json.loads(post_table_json)

        # Establish the connection
        conn = psycopg2.connect(**db_params)
        print("Database connection successful!")
        return conn

    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None