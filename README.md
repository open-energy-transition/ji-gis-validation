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
Before running any scripts, ensure all required dependencies are set up using the Conda package manager. To install the dependencies, run:
```bash
conda env create -f environment.yaml
```
Then, activate the environment using the following command:
```bash
conda activate ji-gis
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

To calculate network parameters (e.g. total system costs, generation mix, and etc) for all countries and planning horizons specified in `database_fill` section of `config.yaml` and upload data into the database, run:
``` bash
snakemake -call fill_main_data_all
```
To calculate network parameters to specific country and horizon, run:
``` bash
snakemake -call results/database_fill/done_AU_2021.txt
```

After filling the main data, it is necessary to estimate cross-horizon information, such as investment needed and investments per CO<sub>2</sub> reduced. To calculate such data, run `fill_investment_co2_all` rule as follows:
``` bash
snakemake -call fill_investment_co2_all
```
Finally, GIS related grid data (e.g. buses, lines, and etc) needs to be uploaded to database. To upload grid data, run:
```bash
snakemake -call fill_grid_data
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
|`co2_emissions`            |Provides CO2 emissions in tCO<sub>2, eq</sub>                  |
|`investments_needed`       |Provides investment needed by carrier to reach net 0 in 2050 in EUR|
|`investment_per_co2_reduced`|Provides average investments required per 1 tonn of CO<sub>2</sub> reduced in EUR/tCO<sub>2</sub>|

Each table contains `scenario_id` key which is in form of `{country_code}_{horizon}_{version}` (e.g. `AU_2021_1`). The version means the iteration number.