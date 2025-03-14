"""
NetCDF Power Grid to PostGIS Converter
-------------------------------------

This script converts PyPSA-Earth NetCDF files containing power grid data into PostGIS tables.
It creates separate tables for buses, carriers, generators, lines, loads, storage units, and stores.

Requirements:
- Python 3.8+
- Required packages: xarray, geopandas, sqlalchemy, geoalchemy2, pypsa, numpy, tqdm, pandas
- PostGIS-enabled PostgreSQL database

Usage:
1. Configure your database parameters in the main() function
2. Prepare a list of paths to your .nc files
3. Run the script

Example:
    netcdf_files = [
        "/path/to/your/AU_2021.nc",
        "/path/to/your/BR_2021.nc",
        # Add more .nc files as needed
    ]

Note: Each .nc file should follow the naming convention: 'XX_YYYY.nc' 
where XX is the country code and YYYY is the year.
"""

import xarray as xr
import geopandas as gpd
from sqlalchemy import create_engine, text, inspect
from geoalchemy2 import Geometry
import os
from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point, LineString
import pypsa
import numpy as np
import json
import glob
from _helpers import mock_snakemake, update_config_from_wildcards, PYPSA_RESULTS_DIR, get_logger

logger = get_logger()


def clean_database(engine):
    """Clean existing PyPSA-specific tables from the database."""
    logger.info("Starting targeted database cleanup...")
    
    tables_to_clean = ['buses', 'carriers', 'generators', 'lines', 'loads', 'storage_units', 'stores']
    
    with engine.connect() as connection:
        for table in tables_to_clean:
            logger.info(f"Deleting table: {table}")
            connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        connection.commit()
    
    logger.info("Cleanup complete - removed only PyPSA-specific tables.")


def create_postgis_table(engine, table_name, columns):
    """Create a PostGIS table with the specified columns."""
    if table_name.lower() in ["all", "user", "group"]:
        table_name = f"table_{table_name}"
    
    logger.info(f"Creating table {table_name} in the database...")
    column_defs = []
    for col, dtype in columns.items():
        if col == 'country_code':
            continue 
        col = col.replace(':', '_').replace('.', '_').replace(' ', '_')
        if dtype == 'geometry':
            column_defs.append(f"{col} GEOMETRY")
        elif isinstance(dtype, np.dtype):
            if np.issubdtype(dtype, np.floating):
                column_defs.append(f"{col} FLOAT")
            elif np.issubdtype(dtype, np.integer):
                column_defs.append(f"{col} INTEGER")
            elif np.issubdtype(dtype, np.bool_):
                column_defs.append(f"{col} BOOLEAN")
            else:
                column_defs.append(f"{col} TEXT")
        elif str(dtype).startswith('float'):
            column_defs.append(f"{col} FLOAT")
        elif str(dtype).startswith('int'):
            column_defs.append(f"{col} INTEGER")
        elif str(dtype) == 'bool':
            column_defs.append(f"{col} BOOLEAN")
        else:
            column_defs.append(f"{col} TEXT")
    
    columns_sql = ", ".join(column_defs)
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY, 
        country_code CHAR(2),
        {columns_sql}
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql))
    logger.info(f"Table {table_name} created successfully.")


def load_data_to_postgis(n, engine, country_code):
    """Load network data into PostGIS tables."""
    tables = ['buses', 'carriers', 'generators', 'lines', 'loads', 'storage_units', 'stores']
    
    for table_name in tables:
        logger.info(f"Loading data from {table_name} for country {country_code} into the database...")
        
        df = getattr(n, table_name)
        index_name = df.index.name if df.index.name else table_name.capitalize()
        df = df.reset_index()
        df = df.rename(columns={'index': index_name})
        columns = [index_name] + [col for col in df.columns if col != index_name]
        df = df[columns]
        dtypes = df.dtypes.to_dict()
        
        if table_name == 'buses':
            ac_carrier = ["AC"]
            ac_buses = n.buses.query("carrier in @ac_carrier").index
            df = df[df[index_name].isin(ac_buses)]
        
        df['country_code'] = country_code
        
        if table_name in ['buses', 'lines']:
            if table_name == 'buses':
                df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']) if 'x' in row and 'y' in row else None, axis=1)
            elif table_name == 'lines':
                buses = n.buses[['x', 'y']]
                df = df.merge(buses.add_suffix('_0'), left_on='bus0', right_index=True)
                df = df.merge(buses.add_suffix('_1'), left_on='bus1', right_index=True)
                df['geometry'] = df.apply(lambda row: 
                    LineString([(row['x_0'], row['y_0']), (row['x_1'], row['y_1'])])
                    if all(x in row and pd.notnull(row[x]) for x in ['x_0', 'y_0', 'x_1', 'y_1']) 
                    else None, axis=1)
        
        df.columns = [col.replace(':', '_').replace('.', '_').replace(' ', '_') for col in df.columns]
        
        if 'geometry' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            gdf = df
        
        columns = {**dtypes, 'country_code': 'object', 'geometry': 'geometry'}
        create_postgis_table(engine, table_name, columns)
        
        try:
            if 'geometry' in gdf.columns:
                gdf_with_geometry = gdf[gdf['geometry'].notnull()]
                gdf_without_geometry = gdf[gdf['geometry'].isnull()]
                if not gdf_with_geometry.empty:
                    gdf_with_geometry.to_postgis(table_name, engine, if_exists='append', index=False)
                if not gdf_without_geometry.empty:
                    gdf_without_geometry.to_sql(table_name, engine, if_exists='append', index=False)
            else:
                gdf.to_sql(table_name, engine, if_exists='append', index=False)
        except Exception as e:
            logger.error(f"Error inserting data into table {table_name}: {e}")


def process_netcdf(file_path, engine):
    """Process a single NetCDF file and load its data into the database."""
    logger.info(f"Processing NetCDF file: {file_path}")
    country_code = os.path.basename(file_path).split('_')[0]
    n = pypsa.Network(file_path)
    load_data_to_postgis(n, engine, country_code)


def verify_database_tables(engine):
    """Verify the contents of the created tables."""
    logger.info("\n=== VERIFYING DATABASE TABLES ===")
    
    with engine.connect() as connection:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        
        for table in tables:
            if table not in ['spatial_ref_sys', 'geography_columns', 'geometry_columns', 'raster_columns', 'raster_overviews']:
                count_query = text(f"SELECT COUNT(*) FROM {table}")
                count = connection.execute(count_query).scalar()
                
                sample_query = text(f"SELECT * FROM {table} LIMIT 5")
                sample = connection.execute(sample_query)
                columns = sample.keys()
                
                logger.info(f"\nTable: {table}")
                logger.info(f"Total records: {count}")
                logger.info(f"Columns: {', '.join(columns)}")
                logger.info("\nSample data:")
                for row in connection.execute(sample_query):
                    logger.info(row)
                logger.info("-" * 80)


def main(scenario_names):
    """Main function to run the conversion process."""
    try:
        # Get POST_TABLE as a JSON string
        post_table_json = os.getenv('POST_TABLE')
        if not post_table_json:
            raise ValueError("POST_TABLE is not set in `.env` file.")
        # Configure your database connection here
        db_params = json.loads(post_table_json)
    
        # Create database connection
        engine = create_engine(
            f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
    
        # Clean existing tables
        #clean_database(engine)

        # List your NetCDF files here
        scenario_folders = [PYPSA_RESULTS_DIR + f"/{x}/networks/" for x in scenario_names]
        netcdf_files = [sorted(glob.glob(os.path.join(scenario_folder, "*.nc"))) for scenario_folder in scenario_folders]
        netcdf_files = [x[0] for x in netcdf_files]

        # Process each NetCDF file
        for netcdf_file in tqdm(netcdf_files, desc="Processing NetCDF files"):
            process_netcdf(netcdf_file, engine)
        
        # Verify the loaded data
        logger.info("\nVerifying loaded tables...")
        verify_database_tables(engine)

        # write to csv file to mark as complete
        df = pd.DataFrame(scenario_names, columns=["scenario_name"])
        df.to_csv(snakemake.output.scenarios, index=False)

    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None
    

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "fill_grid_data",
            countries=["US", "AU"],
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    country_code = config["database_fill"]["countries"]
    horizon = 2021

    # scenario names
    scenario_names = [f"{x}_{horizon}" for x in country_code]

    # fill grid data for scenario names
    main(scenario_names)