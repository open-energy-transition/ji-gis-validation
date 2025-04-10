"""
Optimized script to create and populate energy comparison validation tables in PostgreSQL.
Adapted from validation.py
"""

import os
import sys
import pypsa
import pandas as pd
import pycountry
import logging
import numpy as np
import json
import glob
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings("ignore")
from _helpers import mock_snakemake, update_config_from_wildcards, connect_to_db, \
                     PYPSA_RESULTS_DIR, DATA_DIR


# Helper functions from validation.py
def get_total_demand(n):
    """Calculate total electricity demand in TWh."""
    demand = n.loads_t.p_set.multiply(n.snapshot_weightings.objective, axis=0).sum().sum() / 1e6
    return demand.round(4)

def get_installed_capacities(n):
    """Get installed generation capacities in GW."""
    gen_capacities = n.generators.groupby("carrier").p_nom.sum()
    storage_capacities = n.storage_units.groupby("carrier").p_nom.sum()
    
    capacities = (pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(4)
    
    if "load" in capacities.index:
        capacities.drop("load", inplace=True)
    
    return capacities

def get_generation_mix(n):
    """Get generation mix in TWh."""
    gen_generation = n.generators_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby(n.generators.carrier, axis=1).sum().sum()
    storage_generation = n.storage_units_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby(n.storage_units.carrier, axis=1).sum().sum()
    generation_mix = pd.concat([gen_generation, storage_generation], axis=0) / 1e6
    return generation_mix.round(4)

def get_network_length(n):
    """Get network length and voltages."""
    length = float(n.lines.length.sum())
    voltage_ratings = [float(v) for v in n.lines.v_nom.astype(float).sort_values().unique()]
    return length, voltage_ratings

def real_network_length(country_code):
    """Get real network length and voltages."""
    line_length = {
        "AU":56192.0, "BR":179297.0, "CN":1604838.0, "CO":29169.0, 
        "DE":35796.0, "IN":706348.0, "IT":75246.0, "MX":109747.0, 
        "NG":25633.0, "US":682812.0, "ZA": 0.0
    }
    voltage_ratings = {
        "AU": [66.0, 132.0, 275.0, 330.0],
        "BR": [138.0, 230.0, 345.0, 440.0, 500.0, 600.0, 750.0, 800.0],
        "CN": [110.0, 220.0, 330.0, 500.0, 750.0, 800.0, 1000.0],
        "CO": [110.0, 115.0, 220.0, 230.0, 500.0],
        "DE": [110.0, 220.0, 380.0],
        "IN": [66.0, 132.0, 220.0, 400.0, 765.0],
        "IT": [132.0, 150.0, 220.0, 380.0],
        "MX": [115.0, 230.0, 400.0],
        "NG": [132.0, 330.0],
        "US": [69.0, 115.0, 138.0, 230.0, 345.0, 500.0, 765.0],
        "ZA": [66.0, 88.0, 132.0, 220.0, 330.0]
    }
    return line_length.get(country_code, 0.0), voltage_ratings.get(country_code, [])

def standardize_carrier_data(pypsa_data, eia_data, ember_data):
    """Standardize carriers between different data sources."""
    # Standard carrier mapping
    carrier_mapping = {
        # PyPSA carriers
        'gas': 'Natural gas',
        'CCGT': 'Natural gas',
        'OCGT': 'Natural gas',
        'coal': 'Coal',
        'lignite': 'Coal',
        'nuclear': 'Nuclear',
        'biomass': 'Biomass',
        'hydro': 'Hydro',
        'ror': 'Hydro',
        'solar': 'Solar',
        'wind': 'Wind',
        'offwind-ac': 'Wind',
        'offwind-dc': 'Wind',
        'onwind': 'Wind',
        'load': 'Load shedding',
        
        # EIA carriers
        'Natural gas': 'Natural gas',
        'Other gases': 'Natural gas',
        'Hydroelectricity': 'Hydro',
        'Hydroelectric pumped storage': 'PHS',
        'Biomass and waste': 'Biomass',
        
        # Ember carriers
        'Gas': 'Natural gas',
        'Bioenergy': 'Biomass',
        'Other Fossil': 'Oil'
    }
    
    # Standardize PyPSA data
    if isinstance(pypsa_data, pd.Series):
        pypsa_data.index = pd.Series(pypsa_data.index).map(carrier_mapping).fillna(pypsa_data.index)
        pypsa_data = pypsa_data.groupby(pypsa_data.index).sum()
    
    # Standardize EIA data
    if isinstance(eia_data, pd.DataFrame):
        eia_data['carrier'] = eia_data['country'].map(carrier_mapping)
        eia_data = eia_data.groupby('carrier').sum()
    
    # Standardize Ember data
    if isinstance(ember_data, pd.DataFrame):
        ember_data.index = ember_data.index.map(carrier_mapping).fillna(ember_data.index)
        ember_data = ember_data.groupby(ember_data.index).sum()
    
    # Create combined DataFrame
    carriers = sorted(set(list(pypsa_data.index) + 
                        (list(eia_data.index) if isinstance(eia_data, pd.DataFrame) else []) +
                        (list(ember_data.index) if isinstance(ember_data, pd.DataFrame) else [])))
    
    comparison = pd.DataFrame(index=carriers)
    comparison['pypsa_model'] = pypsa_data
    
    if isinstance(eia_data, pd.DataFrame):
        comparison['eia'] = eia_data.iloc[:, 0]
    else:
        comparison['eia'] = 0.0
        
    if isinstance(ember_data, pd.DataFrame):
        comparison['ember'] = ember_data.iloc[:, 0]
    else:
        comparison['ember'] = 0.0
    
    # Calculate errors
    comparison['error_wrt_eia'] = np.where(
        comparison['eia'] > 0,
        100 * (comparison['pypsa_model'] - comparison['eia']) / comparison['eia'],
        0.0
    )
    
    comparison['error_wrt_ember'] = np.where(
        comparison['ember'] > 0,
        100 * (comparison['pypsa_model'] - comparison['ember']) / comparison['ember'],
        0.0
    )
    
    comparison = comparison.fillna(0.0)
    comparison.index.name = 'carrier'
    
    return comparison.reset_index()

def get_country_name(country_code):
    """Get country name and three-letter code."""
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name, country.alpha_3 if country else (None, None)
    except Exception as e:
        print(f"Error getting country info for {country_code}: {e}")
        return None, None

# Historical data retrieval functions
def get_eia_historic_demand(country_code, horizon, data_dir):
    """Get historical demand from EIA."""
    try:
        eia_data = pd.read_csv(os.path.join(data_dir, "EIA_demands.csv"))
        eia_data = eia_data.rename(columns={"Unnamed: 1": "country"})
        eia_data["country"] = eia_data["country"].str.strip()
        
        country_name, _ = get_country_name(country_code)
        
        if country_name:
            demand = float(eia_data[eia_data['country'].str.contains(country_name, case=False, na=False)][str(horizon)].iloc[0])
            return demand
        return 0.0
    except Exception as e:
        print(f"Error getting EIA demand for {country_code}: {e}")
        return 0.0

def get_ember_historic_demand(country_code, horizon, data_dir):
    """Get historical demand from Ember."""
    try:
        data = pd.read_csv(os.path.join(data_dir, "ember_yearly_full_release_long_format.csv"))
        _, country_code3 = get_country_name(country_code)
        
        demand = data[
            (data["Year"] == horizon) &
            (data["Country code"] == country_code3) &
            (data["Category"] == "Electricity demand") &
            (data["Subcategory"] == "Demand")
        ]["Value"]
        
        return float(demand.iloc[0]) if len(demand) > 0 else 0.0
    except Exception as e:
        print(f"Error getting Ember demand for {country_code}: {e}")
        return 0.0

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('validation_process.log')
        ]
    )
    return logging.getLogger(__name__)

def create_validation_tables(engine):
    """Create validation tables in the database."""
    table_definitions = {
        'demand_comparison': """
            CREATE TABLE IF NOT EXISTS demand_comparison (
                id SERIAL PRIMARY KEY,
                metric TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'capacity_comparison': """
            CREATE TABLE IF NOT EXISTS capacity_comparison (
                id SERIAL PRIMARY KEY,
                carrier TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'generation_comparison': """
            CREATE TABLE IF NOT EXISTS generation_comparison (
                id SERIAL PRIMARY KEY,
                carrier TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'network_comparison': """
            CREATE TABLE IF NOT EXISTS network_comparison (
                id SERIAL PRIMARY KEY,
                metric TEXT,
                pypsa_model FLOAT,
                cited_sources FLOAT,
                error_percent FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """
    }
    
    with engine.connect() as conn:
        for table_name, create_statement in table_definitions.items():
            conn.execute(text(create_statement))
            conn.commit()

def get_eia_historic_capacities(country_code, horizon, data_dir):
    """Get historical capacities from EIA in GW."""
    try:
        # Load EIA data
        eia_data = pd.read_csv(os.path.join(data_dir, "EIA_installed_capacities.csv"))
        eia_data = eia_data.rename(columns={"Unnamed: 1": "country"})
        eia_data["country"] = eia_data["country"].str.strip()
        
        # Get country info
        country_name, _ = get_country_name(country_code)
        
        # Find country data
        if country_name:
            country_index = eia_data[eia_data['country'].str.contains(country_name, case=False, na=False)].index[0]
            capacities = eia_data.iloc[country_index+1:country_index+14][["country", str(horizon)]]
            return capacities
        return None
        
    except Exception as e:
        print(f"Error getting EIA capacities for {country_code}: {e}")
        return None

def get_eia_historic_generation(country_code, horizon, data_dir):
    """Get electricity generation in TWh from EIA."""
    try:
        # Load EIA data
        eia_data = pd.read_csv(os.path.join(data_dir, "EIA_electricity_generation.csv"))
        eia_data.rename(columns={"Unnamed: 1":"country"}, inplace=True)
        eia_data["country"] = eia_data["country"].str.strip()
        
        # Get country name and three-letter code
        country_name, _ = get_country_name(country_code)
        
        # Find generation values for given country
        if country_name and country_name in eia_data.country.unique():
            country_index = eia_data.query("country == @country_name").index[0]
            generation = eia_data.iloc[country_index+1:country_index+18][["country", str(horizon)]]
            return generation
        return None
        
    except Exception as e:
        print(f"Error getting EIA generation for {country_code}: {e}")
        return None

def get_ember_historic_capacities(country_code, horizon, data_dir):
    """Get historical capacities from Ember in GW."""
    try:
        # Load Ember data
        data = pd.read_csv(os.path.join(data_dir, "ember_yearly_full_release_long_format.csv"))
        _, country_code3 = get_country_name(country_code)
        
        capacity_ember = data[
            (data["Country code"] == country_code3) &
            (data["Year"] == horizon) &
            (data["Category"] == "Capacity") &
            (data["Subcategory"] == "Fuel")
        ][["Variable", "Value"]]
        
        # Convert to DataFrame with correct index
        capacity_ember = capacity_ember.set_index("Variable")
        
        # Specific mapping for Ember
        ember_mapping = {
            "Gas": "Fossil fuels",
            "Coal": "Fossil fuels",
            "Other Fossil": "Fossil fuels",
            "Bioenergy": "Biomass",
            "Hydro": "Hydro",
            "Nuclear": "Nuclear",
            "Solar": "Solar",
            "Wind": "Wind"
        }
        
        # Apply mapping and sum values for grouped carriers
        capacity_ember.index = capacity_ember.index.map(lambda x: ember_mapping.get(x, x))
        capacity_ember = capacity_ember.groupby(capacity_ember.index).sum()
        
        # Ensure we have all required carriers
        required_carriers = ["Nuclear", "Fossil fuels", "Hydro", "PHS", "Solar", 
                           "Wind", "Biomass", "Geothermal", "Total capacity"]
        for carrier in required_carriers:
            if carrier not in capacity_ember.index:
                capacity_ember.loc[carrier] = 0.0
                
        # Calculate total
        capacity_ember.loc["Total capacity"] = capacity_ember.drop("Total capacity", errors='ignore').sum()
        
        return capacity_ember
        
    except Exception as e:
        print(f"Error getting Ember capacities for {country_code}: {e}")
        return None

def get_ember_historic_generation(country_code, horizon, data_dir):
    """Get historical generation from Ember in TWh."""
    try:
        # Load Ember data
        data = pd.read_csv(os.path.join(data_dir, "ember_yearly_full_release_long_format.csv"))
        _, country_code3 = get_country_name(country_code)
        
        generation_ember = data[
            (data["Country code"] == country_code3) &
            (data["Year"] == horizon) &
            (data["Category"] == "Electricity generation") &
            (data["Subcategory"] == "Fuel") &
            (data["Unit"] == "TWh")
        ][["Variable", "Value"]].set_index("Variable")
        
        # Standardize fuel types
        generation_ember = generation_ember.rename({
            "Gas": "Natural gas",
            "Bioenergy": "Biomass",
            "Other Fossil": "Oil"
        })
        
        return generation_ember
        
    except Exception as e:
        print(f"Error getting Ember generation for {country_code}: {e}")
        return None

def clean_database(engine):
    """Clean and create comparison tables in the database."""
    print("Starting database cleanup...")
    
    table_definitions = {
        'demand_comparison': """
            CREATE TABLE demand_comparison (
                id SERIAL PRIMARY KEY,
                metric TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'capacity_comparison': """
            CREATE TABLE capacity_comparison (
                id SERIAL PRIMARY KEY,
                carrier TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'generation_comparison': """
            CREATE TABLE generation_comparison (
                id SERIAL PRIMARY KEY,
                carrier TEXT,
                pypsa_model FLOAT,
                eia FLOAT,
                ember FLOAT,
                error_wrt_eia FLOAT,
                error_wrt_ember FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """,
        'network_comparison': """
            CREATE TABLE network_comparison (
                id SERIAL PRIMARY KEY,
                metric TEXT,
                pypsa_model FLOAT,
                cited_sources FLOAT,
                error_percent FLOAT,
                country_code TEXT,
                horizon INTEGER,
                scenario_id TEXT
            )
        """
    }
    
    try:
        with engine.connect() as conn:
            # Temporarily disable foreign key constraints
            conn.execute(text("SET session_replication_role = 'replica';"))
            
            # Drop existing tables if they exist
            for table_name in table_definitions.keys():
                print(f"Dropping table: {table_name}")
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                    conn.commit()
                except Exception as e:
                    print(f"Warning while dropping {table_name}: {e}")
            
            # Wait a moment to ensure tables are dropped
            conn.execute(text("SELECT pg_sleep(0.5)"))
            
            # Create new tables
            for table_name, create_statement in table_definitions.items():
                print(f"Creating table: {table_name}")
                try:
                    conn.execute(text(create_statement))
                    conn.commit()
                except Exception as e:
                    print(f"Error creating table {table_name}: {e}")
                    raise
            
            # Re-enable foreign key constraints
            conn.execute(text("SET session_replication_role = 'origin';"))
            conn.commit()
                
        print("Database cleaned and tables created.")
        
    except Exception as e:
        print(f"Error cleaning database: {e}")
        raise

def clean_scenario_data(scenario_id, engine):
    """Clean scenario data from the database."""
    try:
        with engine.connect() as conn:
            # Delete records for the given scenario_id
            conn.execute(text(f"DELETE FROM demand_comparison WHERE scenario_id = '{scenario_id}'"))
            conn.execute(text(f"DELETE FROM capacity_comparison WHERE scenario_id = '{scenario_id}'"))
            conn.execute(text(f"DELETE FROM generation_comparison WHERE scenario_id = '{scenario_id}'"))
            conn.execute(text(f"DELETE FROM network_comparison WHERE scenario_id = '{scenario_id}'"))
            conn.commit()

        print(f"Scenario data for {scenario_id} cleaned.")

    except Exception as e:
        print(f"Error cleaning scenario data: {e}")
        raise

def compare_demands(network_demand, eia_historic_demand, ember_historic_demand):
    """Compare demands between PyPSA model and historical data."""
    demand = pd.DataFrame(columns=["PyPSA Model", "EIA", "Ember"])
    demand.loc["Electricity Demand [TWh]", ["PyPSA Model", "EIA", "Ember"]] = [
        network_demand, 
        eia_historic_demand, 
        ember_historic_demand
    ]
    demand["Error wrt EIA (%)"] = (100*(demand["PyPSA Model"] - demand["EIA"])/demand["EIA"]).astype(float)
    demand["Error wrt Ember (%)"] = (100*(demand["PyPSA Model"] - demand["Ember"])/demand["Ember"]).astype(float)
    return demand.round(2)

def compare_capacities(network_capacities, eia_historic_capacities, ember_historic_capacities):
    """Compare capacities between PyPSA model and historical data."""
    # Process EIA data
    if isinstance(eia_historic_capacities, pd.DataFrame):
        eia_historic_capacities["country"] = eia_historic_capacities.country.str.replace("(million kW)","").str.strip()
        eia_historic_capacities.set_index("country", inplace=True)
        eia_historic_capacities.columns = ["EIA"]
        eia_historic_capacities.rename(index={
            "Capacity":"Total capacity", 
            "Hydroelectricity":"Hydro", 
            "Biomass and waste":"Biomass", 
            "Hydroelectric pumped storage":"PHS"
        }, inplace=True)
        eia_historic_capacities = eia_historic_capacities.loc[
            ["Nuclear", "Fossil fuels", "Hydro", "PHS", "Solar", "Wind", "Biomass", "Geothermal", "Total capacity"], :
        ]
    
    # Process Ember data
    if isinstance(ember_historic_capacities, pd.DataFrame):
        ember_historic_capacities.columns = ["Ember"]
        ember_historic_capacities.loc["Total capacity", :] = ember_historic_capacities.sum()
    
    # Process PyPSA data
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT", "hydro", "ror", "PHS", "solar", "offwind-ac", "offwind-dc", "onwind", "biomass", "geothermal"]
    network_capacities = network_capacities.reindex(all_carriers, fill_value=0)
    network_capacities.rename(index={
        "nuclear":"Nuclear", "solar":"Solar", "biomass":"Biomass", "geothermal":"Geothermal"
    }, inplace=True)
    
    network_capacities["Fossil fuels"] = network_capacities[["coal", "lignite", "CCGT", "OCGT"]].sum()
    network_capacities["Hydro"] = network_capacities[["hydro", "ror"]].sum()
    network_capacities["Wind"] = network_capacities[["offwind-ac", "offwind-dc", "onwind"]].sum()
    network_capacities = network_capacities.loc[["Nuclear", "Fossil fuels", "Hydro", "PHS", "Solar", "Wind", "Biomass", "Geothermal"]]
    network_capacities["Total capacity"] = network_capacities.sum()
    network_capacities.name = "PyPSA Model"
    network_capacities = network_capacities.to_frame()
    
    # Combine data
    capacities = pd.concat([network_capacities, eia_historic_capacities, ember_historic_capacities], axis=1).astype(float)
    capacities["Error wrt EIA (%)"] = (100*(capacities["PyPSA Model"] - capacities["EIA"])/capacities["EIA"]).round(2)
    capacities["Error wrt Ember (%)"] = (100*(capacities["PyPSA Model"] - capacities["Ember"])/capacities["Ember"]).round(2)
    capacities.fillna(0, inplace=True)
    capacities.index.name = "Capacities [GW]"
    
    return capacities

def compare_generation(network_generation, eia_historic_generation, ember_historic_generation):
    """Compare generation between PyPSA model and historical data."""
    # Process EIA data
    if isinstance(eia_historic_generation, pd.DataFrame):
        eia_historic_generation["country"] = eia_historic_generation.country.str.replace("(billion kWh)","").str.strip()
        eia_historic_generation.set_index("country", inplace=True)
        eia_historic_generation.columns = ["EIA"]
        eia_historic_generation = eia_historic_generation.astype(float)
        eia_historic_generation.rename(index={
            "Generation":"Total generation", 
            "Hydroelectricity":"Hydro", 
            "Biomass and waste":"Biomass", 
            "Hydroelectric pumped storage":"PHS"
        }, inplace=True)
        
        # Add Load shedding if it doesn't exist
        if "Load shedding" not in eia_historic_generation.index:
            eia_historic_generation.loc["Load shedding"] = 0.0
            
        # Sum Natural gas and Other gases if they exist
        if all(gas in eia_historic_generation.index for gas in ["Natural gas", "Other gases"]):
            eia_historic_generation.loc["Natural gas"] = eia_historic_generation.loc[["Natural gas", "Other gases"]].sum()
            
        # Ensure all required carriers exist
        required_carriers = ["Nuclear", "Coal", "Natural gas", "Oil", "Hydro", "PHS", 
                           "Solar", "Wind", "Biomass", "Geothermal", "Load shedding", "Total generation"]
        for carrier in required_carriers:
            if carrier not in eia_historic_generation.index:
                eia_historic_generation.loc[carrier] = 0.0
                
        eia_historic_generation = eia_historic_generation.reindex(required_carriers)
    
    # Process Ember data
    if isinstance(ember_historic_generation, pd.DataFrame):
        ember_historic_generation.columns = ["Ember"]
        # Add Load shedding if it doesn't exist
        if "Load shedding" not in ember_historic_generation.index:
            ember_historic_generation.loc["Load shedding"] = 0.0
        ember_historic_generation.loc["Total generation",:] = ember_historic_generation.sum()
    
    # Process PyPSA data
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT", "oil", "hydro", "ror", "PHS", 
                   "solar", "offwind-ac", "offwind-dc", "onwind", "biomass", "geothermal", "load"]
    network_generation = network_generation.reindex(all_carriers, fill_value=0)
    network_generation.rename(index={
        "nuclear":"Nuclear", "oil":"Oil", "solar":"Solar", "biomass":"Biomass", 
        "geothermal":"Geothermal", "load":"Load shedding"
    }, inplace=True)
    
    network_generation["Coal"] = network_generation[["coal", "lignite"]].sum()
    network_generation["Natural gas"] = network_generation[["CCGT", "OCGT"]].sum()
    network_generation["Hydro"] = network_generation[["hydro", "ror"]].sum()
    network_generation["Wind"] = network_generation[["offwind-ac", "offwind-dc", "onwind"]].sum()
    network_generation["Load shedding"] /= 1e3
    
    required_carriers = ["Nuclear", "Coal", "Natural gas", "Oil", "Hydro", "PHS", 
                        "Solar", "Wind", "Biomass", "Geothermal", "Load shedding"]
    network_generation = network_generation.reindex(required_carriers, fill_value=0)
    network_generation["Total generation"] = network_generation.sum()
    network_generation.name = "PyPSA Model"
    network_generation = network_generation.to_frame()
    
    # Combine data
    generation = pd.concat([network_generation, eia_historic_generation, ember_historic_generation], axis=1).astype(float)
    
    # Calculate errors only where historical data exists
    generation["Error wrt EIA (%)"] = 0.0
    generation["Error wrt Ember (%)"] = 0.0
    
    mask_eia = generation["EIA"] > 0
    generation.loc[mask_eia, "Error wrt EIA (%)"] = (
        100 * (generation.loc[mask_eia, "PyPSA Model"] - generation.loc[mask_eia, "EIA"]) 
        / generation.loc[mask_eia, "EIA"]
    ).round(2)
    
    mask_ember = generation["Ember"] > 0
    generation.loc[mask_ember, "Error wrt Ember (%)"] = (
        100 * (generation.loc[mask_ember, "PyPSA Model"] - generation.loc[mask_ember, "Ember"]) 
        / generation.loc[mask_ember, "Ember"]
    ).round(2)
    
    generation.fillna(0, inplace=True)
    
    return generation
  
def compare_network_lines(network_length, network_voltages, real_length, real_voltages):
    """Compare network lines between PyPSA model and real data."""
    voltage_ratings = sorted(list(set(network_voltages) | set(real_voltages)))
    df_network = pd.DataFrame(index=["Line length [km]"]+voltage_ratings)
    df_network.loc["Line length [km]", ["PyPSA Model", "Cited Sources"]] = [network_length, real_length]
    df_network.loc["Line length [km]", "Error (%)"] = float(100*(network_length - real_length)/real_length) if real_length > 0 else 0.0
    df_network.loc[network_voltages, "PyPSA Model"] = "+"
    df_network.loc[real_voltages, "Cited Sources"] = "+"
    return df_network
  
def process_and_save_to_excel(n, country_code, horizon, data_dir, output_file):
    """Process data and save to Excel as done in validation.py."""
    try:
        # Get all necessary data
        demand = get_total_demand(n)
        generation_capacities = get_installed_capacities(n)
        generation_mix = get_generation_mix(n)
        network_length, network_voltages = get_network_length(n)
        
        # Get historical data
        eia_historic_demand = get_eia_historic_demand(country_code, horizon, data_dir)
        ember_historic_demand = get_ember_historic_demand(country_code, horizon, data_dir)
        
        eia_historic_capacities = get_eia_historic_capacities(country_code, horizon, data_dir)
        ember_historic_capacities = get_ember_historic_capacities(country_code, horizon, data_dir)
        
        eia_historic_generation = get_eia_historic_generation(country_code, horizon, data_dir)
        ember_historic_generation = get_ember_historic_generation(country_code, horizon, data_dir)
        
        real_length, real_voltages = real_network_length(country_code)
        
        # Perform comparisons using validation.py functions
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Compare demand
            demand_comparison = compare_demands(demand, eia_historic_demand, ember_historic_demand)
            demand_comparison.to_excel(writer, sheet_name=country_code, startrow=0, startcol=0)
            
            # Compare capacities
            capacity_comparison = compare_capacities(generation_capacities, eia_historic_capacities, ember_historic_capacities)
            capacity_comparison.to_excel(writer, sheet_name=country_code, startrow=3, startcol=0)
            
            # Compare generation
            generation_comparison = compare_generation(generation_mix, eia_historic_generation, ember_historic_generation)
            generation_comparison.to_excel(writer, sheet_name=country_code, startrow=0, startcol=7)
            
            # Compare network
            network_comparison = compare_network_lines(network_length, network_voltages, real_length, real_voltages)
            network_comparison.to_excel(writer, sheet_name=country_code, startrow=0, startcol=14)
        
        return output_file
        
    except Exception as e:
        print(f"Error processing data for {country_code}: {e}")
        raise

def load_excel_to_db(excel_file, engine, country_code, horizon, version):
    """Load Excel data into database."""
    try:
        # Read data from Excel
        df = pd.read_excel(excel_file, sheet_name=country_code)
        
        # Get scenario_id
        scenario_id = f"{country_code}_{horizon}_{version}"

        # 1. Process demand (first row)
        demand_data = pd.DataFrame({
            'metric': ['Electricity Demand [TWh]'],
            'pypsa_model': [float(df.iloc[0, 1])],  # PyPSA Model
            'eia': [float(df.iloc[0, 2])],         # EIA
            'ember': [float(df.iloc[0, 3])],       # Ember
            'error_wrt_eia': [float(df.iloc[0, 4])],    # Error wrt EIA (%)
            'error_wrt_ember': [float(df.iloc[0, 5])],  # Error wrt Ember (%)
            'country_code': [country_code],
            'horizon': [horizon],
            'scenario_id': [scenario_id]
        })
        
        # 2. Process capacities
        # Find the index where capacities begin
        capacity_start = df.index[df.iloc[:, 0].str.contains('Capacities \[GW\]', na=False)].item()
        capacity_data = []
        
        # Read the next 9 rows containing capacity data
        for i in range(capacity_start + 1, capacity_start + 10):
            row = df.iloc[i]
            capacity_data.append({
                'carrier': row.iloc[0],
                'pypsa_model': float(row.iloc[1]),
                'eia': float(row.iloc[2]),
                'ember': float(row.iloc[3]),
                'error_wrt_eia': float(row.iloc[4]),
                'error_wrt_ember': float(row.iloc[5]),
                'country_code': country_code,
                'horizon': horizon,
                'scenario_id': scenario_id
            })
        
        capacity_df = pd.DataFrame(capacity_data)
        
        # 3. Process generation
        generation_data = []
        carriers = ['Nuclear', 'Coal', 'Natural gas', 'Oil', 'Hydro', 'PHS', 
                   'Solar', 'Wind', 'Biomass', 'Geothermal', 'Load shedding', 'Total generation']
        
        # Read the 12 generation rows (columns 8-13)
        for i, carrier in enumerate(carriers):
            row = df.iloc[i, 7:13]  # Columns 8-13 (indices 7-12)
            generation_data.append({
                'carrier': carrier,
                'pypsa_model': float(row.iloc[1]),
                'eia': float(row.iloc[2]),
                'ember': float(row.iloc[3]),
                'error_wrt_eia': float(row.iloc[4]),
                'error_wrt_ember': float(row.iloc[5]),
                'country_code': country_code,
                'horizon': horizon,
                'scenario_id': scenario_id
            })
        
        generation_df = pd.DataFrame(generation_data)
        
        # 4. Process network (only first row with numeric values)
        network_data = pd.DataFrame({
            'metric': ['Line length [km]'],
            'pypsa_model': [float(df.iloc[0, 15])],      # PyPSA Model
            'cited_sources': [float(df.iloc[0, 16])],    # Cited Sources
            'error_percent': [float(df.iloc[0, 17])],    # Error (%)
            'country_code': [country_code],
            'horizon': [horizon],
            'scenario_id': [scenario_id]
        })
        
        # Handle infinite and NaN values
        for df_to_clean in [demand_data, capacity_df, generation_df, network_data]:
            for col in df_to_clean.select_dtypes(include=[np.number]).columns:
                df_to_clean[col] = df_to_clean[col].replace([np.inf, -np.inf], 0)
                df_to_clean[col] = df_to_clean[col].fillna(0)
        
        # Clean data for the scenario_id
        clean_scenario_data(scenario_id, engine)

        # Save to database
        demand_data.to_sql('demand_comparison', engine, if_exists='append', index=False)
        capacity_df.to_sql('capacity_comparison', engine, if_exists='append', index=False)
        generation_df.to_sql('generation_comparison', engine, if_exists='append', index=False)
        network_data.to_sql('network_comparison', engine, if_exists='append', index=False)
        
        print(f"Successfully loaded data for {country_code}")
        
    except Exception as e:
        print(f"Error loading Excel to database for {country_code}: {e}")
        raise

def main(scenario_name):
    """Main execution function."""
    logger = setup_logging()
    
    # Get POST_TABLE as a JSON string
    post_table_json = os.getenv('POST_TABLE')
    if not post_table_json:
        raise ValueError("POST_TABLE is not set in `.env` file.")
    # Configure your database connection here
    db_params = json.loads(post_table_json)
    
    
    try:
        # Create connection
        engine = create_engine(
            f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
        # Clean and create tables
        logger.info("Cleaning and creating tables...")
        #clean_database(engine)
        
        # Paths to directories
        network_dir = PYPSA_RESULTS_DIR + f"/{scenario_name}/networks"
        
        logger.info(f"Processing networks from: {network_dir}")
        logger.info(f"Using data from: {DATA_DIR}")
        logger.info(f"Saving results to: {snakemake.output.excel}")
        
        # Verify directories exist
        if not os.path.exists(network_dir):
            raise FileNotFoundError(f"Networks directory not found: {network_dir}")
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
            
        for file in os.listdir(network_dir):
            if file.endswith('.nc'):
                try:
                    # Load network
                    network_path = os.path.join(network_dir, file)
                    logger.info(f"Loading network from: {network_path} for scenario {scenario_name}")
                    n = pypsa.Network(network_path)
                    
                    # Process and save to Excel
                    excel_file = process_and_save_to_excel(n, country_code, horizon, DATA_DIR, snakemake.output.excel)
                    logger.info(f"Excel file created: {excel_file}")
                    
                    # Load Excel to database
                    load_excel_to_db(excel_file, engine, country_code, horizon, version)
                    logger.info(f"Data loaded to database for {country_code}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
                    logger.exception("Stack trace:")
                    continue
                    
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.exception("Stack trace:")
        sys.exit(1)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "fill_statistics",
            countries="AU",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    country_code = config["database_fill"]["countries"]
    horizon = snakemake.params.planning_horizon[0]  # horizon for historical data
    version = config["database_fill"]["version"]

    # scenario name
    scenario_name = f"{country_code}_{horizon}"

    main(scenario_name)