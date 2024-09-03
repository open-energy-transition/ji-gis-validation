# JI-GIS data validation

## Instruction 
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