import glob
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import xskillscore as xs


def extract_closest_co2vmr_data(
    co2vmr: float, mt: int, var: str, FNS: List[str]
) -> Tuple[xr.Dataset, List[int]]:
    """
    Extract data for the given variable where CO₂ VMR is closest to the target value for each file.

    Parameters:
    - co2vmr: The target CO₂ VMR value (float).
    - var: The variable name to extract (str).
    - FNS: List of file paths to NetCDF datasets (list of strings).

    Returns:
    - xarray.DataArray: DataArray containing the stacked data across all files, with dimensions 'time', 'lat', and 'lon'.
    """
    dats_list = []
    closest_index_list = []
    for fn in FNS:
        # Open the dataset
        CESM_LE = xr.open_dataset(fn)

        # CESM files are 1 month forward!
        feb_data = CESM_LE.where(CESM_LE["time.month"] == mt, drop=True)

        # Get the CO2 VMR values
        co2vmr_values = feb_data["co2vmr"].values

        # Calculate the absolute difference between the current CO₂ VMR values and the target
        abs_diff = np.abs(co2vmr_values - co2vmr)

        # Find the index of the minimum difference
        closest_index = np.argmin(abs_diff)
        closest_index_list.append(closest_index)
        # Extract the corresponding data point for the given variable at the closest CO₂ VMR
        dats_do = feb_data.isel(time=closest_index)[var].values

        # Append this data to the list
        dats_list.append(dats_do)

    # Stack the list of data arrays into a new dimension ('time')
    dats_stack = np.stack(dats_list, axis=0)

    # Get lat, lon, and time coordinates (assuming they are consistent across all files)
    lat = feb_data["lat"].values
    lon = feb_data["lon"].values
    times = np.arange(len(FNS))  # Length = 50 (members 0-49)

    # Create an xarray DataArray from the stacked data
    dats_da = xr.DataArray(
        dats_stack,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon},
    )

    return dats_da.to_dataset(name=var), closest_index_list


def closest_co2vmr_year(co2vmr: float, FNS: List[str]) -> Tuple[int, int, str]:

    closest_diff = np.inf  # Initialize with a large value
    closest_time = None
    closest_file = None

    # Dictionary to store the time data from each file
    dict_times = {}

    for fn in FNS:
        # Open the dataset once per file
        CESM_LE = xr.open_dataset(fn)
        timers = CESM_LE["time"].values
        co2vmr_values = CESM_LE["co2vmr"].values

        # Store the time information in a dictionary for later retrieval
        dict_times[fn] = timers

        # Calculate the absolute difference for the CO₂ VMR values in this file
        abs_diff = np.abs(co2vmr_values - co2vmr)

        # Find the index of the minimum difference in this file
        min_index = np.argmin(abs_diff)

        # Update if the current file has a closer CO₂ VMR value
        if abs_diff[min_index] < closest_diff:
            closest_diff = abs_diff[min_index]
            closest_time = timers[min_index]
            closest_file = fn
            print("min dist:", abs_diff[min_index])

    if closest_time is None or closest_file is None:
        raise ValueError("No suitable CO₂ VMR match found in the provided files.")
    # Return the year, month, and file name of the closest match
    return closest_time.year, closest_time.month, closest_file


def concat_files_by_month(month_get: int, file_pattern: str) -> xr.Dataset:
    """
    Concatenate datasets along a 'samples' dimension, selecting data from a specific month.

    Parameters:
    month_get (int): The month to filter by (e.g., 2 for February).
    file_pattern (str): The glob pattern for the input NetCDF files.

    Returns:
    xarray.Dataset: Concatenated dataset along the 'samples' dimension.
    """
    # Step 1: Get the list of files
    FNS = sorted(glob.glob(file_pattern))

    # Step 2: Create an empty list to hold the filtered datasets
    datasets = []

    # Track the running count of samples
    sample_counter = 0

    # Step 3: Loop over each file and load the dataset, selecting only data for the given month
    for fps in FNS:
        # Open the dataset
        DS = xr.open_dataset(fps)

        # Select only data from the specified month
        mo_g = month_get + 1

        if mo_g == 13:
            mo_g = 1

        DS_feb = DS.sel(time=DS["time.month"] == mo_g)

        # Determine the number of time steps in the current dataset
        n_samples = DS_feb.sizes["time"]

        # Replace the 'time' dimension with 'samples' and increment the sample index
        DS_feb = DS_feb.rename({"time": "samples"})  # Rename 'time' to 'samples'

        # Assign a new 'samples' coordinate, starting from the running sample counter
        DS_feb = DS_feb.assign_coords(samples=(sample_counter + np.arange(n_samples)))

        # Update the sample counter for the next dataset
        sample_counter += n_samples

        # Append the modified dataset to the list
        datasets.append(DS_feb)

    # Step 4: Concatenate along the new "samples" dimension
    DS_concat = xr.concat(datasets, dim="samples")

    return DS_concat


def get_data(momo: int, flag: bool = False) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, str],
    Dict[int, List[str]],
    Dict[int, List[int]],
]:
    FNS = sorted(
        glob.glob(
            f"//users_home/training/data/Group_01/Ensembles/samples_governance_indexes_3944_month{momo:02}*.nc"
        )
    )

    mea_trend = []
    std_trend = []
    all_mea_trend = {}
    all_mea_file_paths = {}

    lens_mea_trend = []
    lens_std_trend = []
    all_lens_mea_trend = {}
    all_lens_file_paths = {}
    all_lens_closest_indexes = {}

    for ee, fn in enumerate(FNS):
        print("doing:", ee + 2015, fn)
        DS = xr.open_dataset(fn)
        if flag == True:
            DS = DS.sel(lat=slice(-5, 5), lon=slice(190, 240))  # lon=slice(120,170)

        lat = DS["lat"]
        # Calculate weights: cos(lat) in radians
        weights = np.cos(np.deg2rad(lat))

        tmean = DS["TREFHT"].weighted(weights).mean(dim=["lat", "lon", "samples"])
        mea_trend.append(tmean.values)

        all_mea_trend[ee] = (
            DS["TREFHT"].weighted(weights).mean(dim=["lat", "lon"]).values
        )
        all_mea_file_paths[ee] = fn

        DDS = xs.resample_iterations_idx(DS["TREFHT"], 50, "samples")
        std_trend.append(
            DDS.sel(samples=slice(0, 50))
            .weighted(weights)
            .mean(dim=["samples", "lat", "lon"])
            .std("iteration")
        )

        # now for the LENS:
        # 1. find files:
        co2vmr = float(fn.split("co2")[-1].split("_")[0])
        print(co2vmr)
        FNS = sorted(
            glob.glob(f"//users_home/training/data/Group_01/LENS/*1001.001*.nc")
        )

        yt, mt, fout = closest_co2vmr_year(co2vmr, FNS)

        if mt != ((momo + 1) % 12):
            print("NOOOOOO!--- an error has occurred")

        FNS_year = sorted(
            glob.glob(
                f'//users_home/training/data/Group_01/LENS/*.{int(fout.split(".")[-2][:4])}*.nc'
            )
        )

        results_da1, closest_index_list = extract_closest_co2vmr_data(
            co2vmr, mt, "TREFHT", FNS_year
        )

        #### SAVE THE PATHS
        all_lens_file_paths[ee] = FNS_year
        all_lens_closest_indexes[ee] = closest_index_list

        if flag == True:
            results_da1 = results_da1.sel(lat=slice(-5, 5), lon=slice(190, 240))

        tmean_lens = (
            results_da1["TREFHT"].weighted(weights).mean(dim=["lat", "lon", "time"])
        )
        lens_mea_trend.append(tmean_lens.values)

        all_lens_mea_trend[ee] = (
            results_da1["TREFHT"].weighted(weights).mean(dim=["lat", "lon"]).values
        )
        DDS_lens = xs.resample_iterations_idx(results_da1["TREFHT"], 50, "time")
        lens_std_trend.append(
            DDS_lens.weighted(weights).mean(dim=["time", "lat", "lon"]).std("iteration")
        )

    return (
        all_mea_trend,
        all_lens_mea_trend,
        all_mea_file_paths,
        all_lens_file_paths,
        all_lens_closest_indexes,
    )


def get_snapshot_Lens(paths, closest_inds, month, picked_year, picked_index):
    mt = (month + 1) % 12
    path = paths[picked_year][picked_index]
    index_to_get = closest_inds[picked_year][picked_index]
    DS = xr.open_dataset(path)
    DS = DS.where(DS["time.month"] == mt, drop=True)
    DS = DS.isel(time=index_to_get)
    return DS


def get_snapshot_Diff(paths, month, picked_year, picked_index):
    DS = xr.open_dataset(paths[picked_year])
    DS = DS.sel(samples=picked_index)
    return DS
