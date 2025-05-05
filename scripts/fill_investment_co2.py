# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pypsa
import pandas as pd
import logging
from _helpers import mock_snakemake, update_config_from_wildcards, connect_to_db, \
                     load_network, get_solved_network_path, get_logger
from fill_main_data import write_to_db

logger = get_logger()


def sql_read_data(table_name, scenario_id):
    sql = f"SELECT * FROM pypsa_earth_db.public.{table_name} WHERE scenario_id = '{scenario_id}' "
    return sql


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "fill_investment_co2", 
            countries="US",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    country_code = config["database_fill"]["countries"]
    horizons = config["database_fill"]["planning_horizon"]
    version = config["database_fill"]["version"]

    # ensure that both 2021 and 2050 are in config
    if all(year in horizons for year in [2021, 2050]):
        logger.info("Both 2021 and 2050 are in horizons.")
    else:
        raise ValueError("Error: Not all required years (2021, 2050) are specified in 'horizons'. Please check your configuration.")
        
    # read co2 emissions and investment data
    df = pd.DataFrame()
    table_names = {
        "investment": "investment_costs_by_techs",
        "co2_emissions": "co2_emissions"
    }
    investments = pd.DataFrame()

    for horizon in horizons:
        # scenario_id
        scenario_id = f"{country_code}_{horizon}_{version}"
        for param, table_name in table_names.items():
            with connect_to_db() as conn:
                temp_df = pd.read_sql(sql_read_data(table_name, scenario_id), conn)
                temp_df = temp_df.set_index("carrier")
            if param == "investment":
                # obtain total investments in billion EURs
                value = temp_df["investment_cost"].sum()
                # obtain investments by carrier in billion EURs
                investments_by_carrier = temp_df["investment_cost"].copy()
                investments_by_carrier.name = horizon
                investments = pd.concat([investments, investments_by_carrier], axis=1)
            elif param == "co2_emissions":
                # obtain total co2 emissions in tCO2_eq
                value = temp_df["co2_emission"].sum()
            # record to df
            df.loc[horizon, param] = value

    # calculate delta co2 emissions in tCO2_eq
    delta_co2 = df.loc[2021, "co2_emissions"] - df.loc[2050, "co2_emissions"]

    # calculate investment needed
    investments.fillna(0, inplace=True)
    investments.index.name = "carrier"
    investment_needed = (investments[2050] - investments[2021]).clip(lower=0).to_frame()
    investment_needed.columns = ["investment_needed"]
    investment_needed.reset_index(inplace=True)
    investment_needed["country_code"] = country_code
    investment_needed["horizon"] = 2050
    investment_needed["scenario_id"] = f"{country_code}_2050_{version}"

    # calculate investments per co2 reduced in EUR/tCO2_eq (investments in billion EURs, co2 in tCO2_eq)
    delta_investment = investment_needed.investment_needed.sum()
    investment_per_co2_reduced = (delta_investment / delta_co2) * 1e9

    # output df (refer to 2050, because investments needed to reach net 0 in 2050)
    df_output = pd.DataFrame(data=[investment_per_co2_reduced], columns=["investment_per_co2_reduced"])
    df_output["country_code"] = country_code
    df_output["horizon"] = 2050
    df_output["scenario_id"] = f"{country_code}_2050_{version}"
    
    # write to Excel file
    with pd.ExcelWriter(snakemake.output.excel, engine="openpyxl") as writer:
        df_output.to_excel(writer, sheet_name="investment_per_co2_reduced", index=False)
        investment_needed.to_excel(writer, sheet_name="investments_needed", index=False)

    # write data to database
    try:
        write_to_db(df_output, conn=connect_to_db, table_name="investment_per_co2_reduced")
        write_to_db(investment_needed, conn=connect_to_db, table_name="investments_needed")
    except Exception as e:
        logger.error(f"Error happened when writing to table investment_per_co2_reduced: {e}")
    