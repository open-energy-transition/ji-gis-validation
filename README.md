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
