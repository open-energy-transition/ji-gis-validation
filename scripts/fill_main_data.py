# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pypsa
import pandas as pd
import logging
from _helpers import mock_snakemake, update_config_from_wildcards, connect_to_db, \
                     load_network, get_solved_network_path, get_logger, PYPSA_RESULTS_DIR, \
                     PREFIX_TO_REMOVE, RENAME_IF_CONTAINS, RENAME_IF_CONTAINS_DICT, RENAME

logger = get_logger()


def compute_costs(n, cost_type):
    assert cost_type in ["Operational", "Capital"], "Type variable must be 'Operational' or 'Capital'"
    costs = n.statistics()[[f"{cost_type} Expenditure"]]
    new_index = [':'.join(idx) for idx in costs.index]
    costs.index = new_index
    costs.columns = ["cost"]
    return costs


def sum_costs(cap_cost_df, op_cost_df):
    total_cost = cap_cost_df.add(op_cost_df, fill_value=0)
    new_index = [x.split(":")[1] for x in total_cost.index]
    total_cost.index = new_index
    return total_cost


def get_total_costs(network, non_generating_carriers):
    cap_costs = compute_costs(network, "Capital")
    op_costs = compute_costs(network, "Operational")
    total_costs = sum_costs(cap_costs, op_costs)
    
    df = total_costs.groupby(total_costs.index).sum()

    # convert to billions
    df = df / 1e9
    df = df.groupby(df.index.map(rename_techs)).sum()
    df.drop("-", inplace=True)

    # drop load shedding
    df.drop("Load shedding", inplace=True, errors='ignore')

    # consider only carriers which have share in generation mix for 2050
    if horizon == '2050':
        # format carriers with 0 generation to be dropped
        non_generating_carriers = [s if s.isupper() else s.capitalize() for s in non_generating_carriers]

        # drop carriers which have 0 generation
        df.drop(index=non_generating_carriers, errors='ignore', inplace=True)

    return df


def get_investment_costs(network, non_generating_carriers):
    cap_costs = compute_costs(network, "Capital")
    new_index = [x.split(":")[1] for x in cap_costs.index]
    cap_costs.index = new_index
    
    df = cap_costs.groupby(cap_costs.index).sum()
    
    # convert to billions
    df = df / 1e9
    df = df.groupby(df.index.map(rename_techs)).sum()
    df.drop("-", inplace=True)

    # consider only carriers which have share in generation mix for 2050
    if horizon == '2050':
        # format carriers with 0 generation to be dropped
        non_generating_carriers = [s if s.isupper() else s.capitalize() for s in non_generating_carriers]

        # drop carriers which have 0 generation
        df.drop(index=non_generating_carriers, errors='ignore', inplace=True)

    return df


def rename_techs(label):

    for ptr in PREFIX_TO_REMOVE:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in RENAME_IF_CONTAINS:
        if rif in label:
            label = rif

    for old, new in RENAME_IF_CONTAINS_DICT.items():
        if old in label:
            label = new

    for old, new in RENAME.items():
        if old == label:
            label = new
    return label


def get_load_shedding_cases(network, threshold=1):
    # get load shedding generators
    load_shed_gens = network.generators.query("carrier == 'load'").index
    # get load shedding dispatch
    load_shed_dispatch = network.generators_t.p.multiply(network.snapshot_weightings.objective, axis=0)[load_shed_gens]
    # get load shedding cases >= threshold in MW
    load_shed_cases = load_shed_dispatch >= threshold * 1e3
    # get mapping of load shedding generators to buses
    load_shed_to_bus_mapping = network.generators.query("carrier == 'load'").bus
    # map load shedding cases to buses
    load_shed_cases = load_shed_cases.rename(columns=load_shed_to_bus_mapping)
    return load_shed_cases


def get_average_electricity_price(network):
    # get load shedding cases (shedding >= 1 MW)
    load_shed_cases = get_load_shedding_cases(network, threshold=1)
    # AC buses
    ac_buses = network.buses.query("carrier == 'AC'").index
    # get electricity load indices
    load_indices = network.loads.query("bus in @ac_buses").index
    # get loads in MWh
    elec_loads = network.loads_t.p_set.multiply(network.snapshot_weightings.objective, axis=0)[load_indices]
    # get total costs in EUR for non load shedding hours
    total_costs = elec_loads.multiply(network.buses_t.marginal_price[elec_loads.columns]).multiply(network.snapshot_weightings.objective, axis=0)[~load_shed_cases].fillna(0).sum().sum()
    # get total load in MWh for non load shedding hours
    total_load = network.loads_t.p_set.multiply(network.snapshot_weightings.objective, axis=0)[load_indices][~load_shed_cases].fillna(0).sum().sum()
    # get costs EUR/MWh
    prices = total_costs / total_load
    return prices.round(2)


def get_generation_mix(n):
    # definde elec buses
    elec = ["AC", "low voltage"]
    elec = n.buses.query("carrier in @elec").index
    
    # elec mix from generators
    gens = n.generators.query("bus in @elec").index
    elec_mix = n.generators_t.p[gens].multiply(n.snapshot_weightings.objective,axis=0).T.groupby(n.generators.carrier).sum().T.sum()
    elec_mix["load"] /= 1e3
    
    # elec mix storage units
    elec_mix_hydro = n.storage_units_t.p.multiply(n.snapshot_weightings.objective,axis=0).T.groupby(n.storage_units.carrier).sum().T.sum()
    
    # elec mix from stores (csp)
    discharger_techs = ["csp"]
    discharger_links = n.links.query("carrier in @discharger_techs").index
    elec_mix_links = -n.links_t.p1[discharger_links].multiply(n.snapshot_weightings.objective,axis=0).T.groupby(n.links.carrier).sum().T.sum()
    
    # concatenate generations
    total_mix = pd.concat([elec_mix, elec_mix_hydro, elec_mix_links], axis=0)
    total_mix.rename(index={"offwind-ac":"offwind", "offwind-dc":"offwind",
                            "load":"load shedding", "ror":"hydro", "lignite":"coal"}, inplace=True)
    # get generation mix in TWh
    total_mix = total_mix.groupby(total_mix.index).sum() / 1e6
    # clip negative generation
    total_mix = total_mix.clip(lower=0)
    # drop load shedding
    total_mix.drop("load shedding", inplace=True, errors='ignore')
    return total_mix


def get_total_load(n):
    # definde elec buses
    elec = ["AC", "low voltage"]
    elec = n.buses.query("carrier in @elec").index
    loads = n.loads.query("bus in @elec").index
    # get total demand in TWh
    demand = n.loads_t.p_set[loads].multiply(n.snapshot_weightings.objective,axis=0).sum().sum() / 1e6
    return demand


def get_installed_capacities(n, non_generating_carriers):
    gen_capacities = n.generators.groupby("carrier").p_nom.sum()
    storage_capacities = n.storage_units.groupby("carrier").p_nom.sum()
    capacities = (pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(4)
    if "load" in n.generators.carrier.unique():
        capacities.drop("load", inplace=True)
    # drop carriers which have 0 generation
    capacities.drop(index=non_generating_carriers, errors='ignore', inplace=True)
    return capacities


def get_optimal_capacities(n, non_generating_carriers):
    gen_capacities = n.generators.groupby("carrier").p_nom_opt.sum()
    storage_capacities = n.storage_units.groupby("carrier").p_nom_opt.sum()
    capacities = (pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(4)
    if "load" in n.generators.carrier.unique():
        capacities.drop("load", inplace=True)
    # drop carriers which have 0 generation
    capacities.drop(index=non_generating_carriers, errors='ignore', inplace=True)
    return capacities


def get_capacity_expansion(optimal_capacity, installed_capacity):
    capacity_expansion = optimal_capacity - installed_capacity
    return capacity_expansion


def get_co2_emissions(n):
    # get generation amount by each carrier in MWh_el
    generation_mix = get_generation_mix(n) * 1e6
    # get efficiency from convertion from thermal to electrical energy
    efficiency_therm_to_elec = n.generators.groupby("carrier").efficiency.mean()
    # get co2 emissions per MWh_el
    co2_emissions_MWh_el = n.carriers.co2_emissions.div(efficiency_therm_to_elec)
    co2_emissions = generation_mix.multiply(co2_emissions_MWh_el, fill_value=0) # in tCO2_eq
    # drop entries with 0 emission and NaN value
    co2_emissions.dropna(inplace=True)

    return co2_emissions


def write_to_db(df, conn, table_name: str):
    """
    Writes a DataFrame to a PostgreSQL table. Deletes existing rows based on `scenario_id` and writes new data.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to write. Must have columns carrier, costs, country_code, horizon, scenario_id.
    - table_name (str): Name of the PostgreSQL table.
    """
    full_table_name = "pypsa_earth_db.public." + table_name
    # Establish the connection
    with conn() as connection:
        try:
            # Extract unique scenario_id values from the DataFrame
            scenario_ids = df['scenario_id'].unique()
            
            # Start a transaction
            with connection.cursor() as cursor:
                # Delete rows for all scenario_ids in the DataFrame
                for scenario_id in scenario_ids:
                    cursor.execute(f"DELETE FROM {full_table_name} WHERE scenario_id = %s", (scenario_id,))
                
                # Commit the deletion transaction
                connection.commit()
                # Add logs about deleting entries
                logger.info(f"Deleted entries from {table_name} with scenario_id: {', '.join(scenario_ids)}")

            # Extract column names from the DataFrame
            columns = list(df.columns)
            columns_placeholder = ", ".join(["%s"] * len(columns))
            columns_names = ", ".join(columns)

            # Write the DataFrame to the table
            with connection.cursor() as cursor:
                rows = df.itertuples(index=False, name=None)
                
                # Dynamically build and execute the INSERT query
                cursor.executemany(
                    f"""
                    INSERT INTO {full_table_name} ({columns_names})
                    VALUES ({columns_placeholder})
                    """,
                    rows
                )
                
                # Commit the insert transaction
                connection.commit()
                # Add logs about insert
                logger.info(f"Insert entries into {table_name} with scenario_id: {', '.join(scenario_ids)}")
        
        except Exception as e:
            connection.rollback()
            raise e


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "fill_main_data", 
            countries="AU",
            planning_horizon="2050",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    country_code = config["database_fill"]["countries"]
    horizon = config["database_fill"]["planning_horizon"]
    version = config["database_fill"]["version"]

    # solved network path
    network = load_network(get_solved_network_path(country_code, horizon, PYPSA_RESULTS_DIR))

    # scenario name
    scenario_name = f"{country_code}_{horizon}_{version}"

    # generation mix in TWh
    generation_mix = get_generation_mix(network).to_frame()
    generation_mix.reset_index(inplace=True)
    generation_mix.rename(columns={"index": "carrier", 0: "generation"}, inplace=True)
    generation_mix["horizon"] = int(horizon)
    generation_mix["country_code"] = country_code
    generation_mix["scenario_id"] = scenario_name
    # get non-generating carriers
    non_generating_carriers = generation_mix[generation_mix["generation"] == 0].carrier.values
    # remove PHS from non-generating carriers
    non_generating_carriers = [s for s in non_generating_carriers if s != "PHS"]

    # total costs by technologies in billion EUR
    total_costs_by_techs = get_total_costs(network, non_generating_carriers)
    total_costs_by_techs.reset_index(inplace=True)
    total_costs_by_techs.rename(columns={"index": "carrier", "cost": "total_costs"}, inplace=True)
    total_costs_by_techs["horizon"] = int(horizon)
    total_costs_by_techs["country_code"] = country_code
    total_costs_by_techs["scenario_id"] = scenario_name

    # investment costs by technology in billion EUR
    investment_costs_by_techs = get_investment_costs(network, non_generating_carriers)
    investment_costs_by_techs.reset_index(inplace=True)
    investment_costs_by_techs.rename(columns={"index": "carrier", "cost": "investment_cost"}, inplace=True)
    investment_costs_by_techs["horizon"] = int(horizon)
    investment_costs_by_techs["country_code"] = country_code
    investment_costs_by_techs["scenario_id"] = scenario_name

    # electricity price in EUR/MWh
    electricity_prices = get_average_electricity_price(network)
    electricity_prices = pd.DataFrame(data=[electricity_prices], columns=["electricity_price"])
    electricity_prices["horizon"] = int(horizon)
    electricity_prices["country_code"] = country_code
    electricity_prices["scenario_id"] = scenario_name

    # total load in TWh
    total_load = get_total_load(network)
    total_load = pd.DataFrame(data=[total_load], columns=["total_load"])
    total_load["horizon"] = int(horizon)
    total_load["country_code"] = country_code
    total_load["scenario_id"] = scenario_name

    # get installed capacity in GW
    installed_capacity = get_installed_capacities(network, non_generating_carriers).to_frame()
    installed_capacity.reset_index(inplace=True)
    installed_capacity.rename(columns={"index": "carrier", "p_nom": "installed_capacity"}, inplace=True)
    installed_capacity["horizon"] = int(horizon)
    installed_capacity["country_code"] = country_code
    installed_capacity["scenario_id"] = scenario_name

    # get optimal capacity in GW
    optimal_capacity = get_optimal_capacities(network, non_generating_carriers).to_frame()
    optimal_capacity.reset_index(inplace=True)
    optimal_capacity.rename(columns={"index": "carrier", "p_nom_opt": "optimal_capacity"}, inplace=True)
    optimal_capacity["horizon"] = int(horizon)
    optimal_capacity["country_code"] = country_code
    optimal_capacity["scenario_id"] = scenario_name

    # get capacity expansion in TWh
    capacity_expansion = get_capacity_expansion(get_optimal_capacities(network, non_generating_carriers), 
                                                get_installed_capacities(network, non_generating_carriers)).to_frame()
    capacity_expansion.reset_index(inplace=True)
    capacity_expansion.rename(columns={"index": "carrier", 0: "capacity_expansion"}, inplace=True)
    capacity_expansion["horizon"] = int(horizon)
    capacity_expansion["country_code"] = country_code
    capacity_expansion["scenario_id"] = scenario_name

    # get co2 emission in tCO2_eq
    co2_emissions = get_co2_emissions(network).to_frame()
    co2_emissions.reset_index(inplace=True)
    co2_emissions.rename(columns={"index": "carrier", 0: "co2_emission"}, inplace=True)
    co2_emissions["horizon"] = int(horizon)
    co2_emissions["country_code"] = country_code
    co2_emissions["scenario_id"] = scenario_name


    # define mapping of table name and data
    table_data_mapping = {
        "total_costs_by_techs": total_costs_by_techs,
        "investment_costs_by_techs": investment_costs_by_techs,
        "electricity_prices": electricity_prices,
        "generation_mix": generation_mix,
        "total_load": total_load,
        "installed_capacity": installed_capacity,
        "optimal_capacity": optimal_capacity,
        "capacity_expansion": capacity_expansion,
        "co2_emissions": co2_emissions
    }

    # write to Excel file
    with pd.ExcelWriter(snakemake.output.excel, engine="openpyxl") as writer:
        for sheet_name, df in table_data_mapping.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # write data to database
    for table, df in table_data_mapping.items():
        try:
            write_to_db(df, conn=connect_to_db, table_name=table)
        except Exception as e:
            logger.error(f"Error happened when writing to table {table}: {e}")
    