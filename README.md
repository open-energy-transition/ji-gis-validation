# JI-GIS data validation

# Instructions and Usage
## 1. Data Structure and Organization
The PyPSA data, containing prenetworks, resources data, and solved networks, should be placed into `pypsa_data` folder. In particular, the structure of `pypsa_data` folder needs to be as follows:

```
pypsa_data/
├── networks/
|   └── AU_2021/
|       └── base.nc
├── resources/
└── results/
    └── AU_2021/
        └── networks/
            └── elec_s_50flex_ec_lcopt_1H.nc
```

## 2. Running validation

This project utilizes [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to automate the execution of validation and plotting scripts, ensuring efficient and reproducible workflows. Configuration settings for *snakemake* are available in the `configs/config.yaml` file.

To run validation across all countries specified in the `config.yaml`, navigate to the working directory (`.../ji-gis-validation/`) and use the following command:
```bash
snakemake -call validate_all
```
* **Note:** Ensure that the PyPSA solved networks, resources, and prenetworks for the countries and horizons specified in the `config.yaml` file are placed in the `pypsa_data` folder, as described in Section 1.

To run validation to a single country (e.g. for Austria in 2021), the following command can be used:
``` bash
snakemake -call results/validation/validation_AU_2021.xlsx
```

## 3. Upload computed data into database

First, it is important to place `.env` file with postgresql credentials into working directory (i.e. `path-to/ji-gis-app/.`). The`.env` file should contain information for database connection, such as host, port, database name, username, and password, and meet the following format:  
``` bash
POST_TABLE = {"dbname": "database_name", "user": "yourusername", "password": "yourpassword", "host": "ipaddress", "port": "yourport"}
```
Replace values with valid arguements. Then install `python-dotenv` and `psycopg2` on top of existing `pypsa-earth` conda environment by running:
``` bash
pip install psycopg2 python-dotenv
```
To calculate network parameters for all countries and planning horizons specified in `database_fill` section of `config.yaml` and upload data into the database, run:
``` bash
snakemake -call fill_main_data_all
```
To calculate network parameters to specific country and horizon, run:
``` bash
snakemake -call results/database_fill/done_AU_2021.txt
```
The list of all calculated parameters for each scenario is:
|Table name                 |Description                                        |
|---------------------------|---------------------------------------------------|  
|`total_costs_by_techs`     |Provides total costs by carrier in billion EUR     |
|`investment_costs_by_techs`|Provides investment costs by carrier in billion EUR|
|`electricity_prices`       |Provides electricity price in EUR/MWh              |
|`generation_mix`           |Provides generation mix in TWh                     |
|`total_load`               |Provides total load in TWh                         |
|`installed_capacity`       |Provides installed capacities by carrier in GW     |
|`optimal_capacity`         |Provides optimal capacities by carrier in GW       |
|`capacity_expansion`       |Provides capacity expansion by carrier in GW       |
|`co2_emissions`            |Provides CO2 emissions in tCO2_eq                  |

Each table contains `scenario_id` key which is in form of `{country_code}_{horizon}_{version}` (e.g. `AU_2021_1`). The version means the iteration number.