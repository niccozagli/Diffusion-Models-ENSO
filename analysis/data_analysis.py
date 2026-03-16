import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import xarray as xr
    from diffusion_models_enso.analysis import (
        compute_nino34_index,
        reconstruct_sample_anomaly,
        select_month,
        select_reference_samples,
        drop_empty_years,
    )
    from diffusion_models_enso.utils import find_repo_root
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeatures

    return (
        Path,
        ccrs,
        cfeatures,
        compute_nino34_index,
        drop_empty_years,
        find_repo_root,
        mo,
        plt,
        reconstruct_sample_anomaly,
        select_month,
        select_reference_samples,
        xr,
    )


@app.cell
def _(Path, find_repo_root):
    repo_folder_path = find_repo_root(Path(__file__))
    data_folder_path = repo_folder_path / "data"
    input_data_folder_path = Path("/Volumes/Nicco")
    return data_folder_path, input_data_folder_path


@app.cell
def _(data_folder_path, xr):
    ds = xr.open_dataset(filename_or_obj=data_folder_path / "diffusion.nc")
    month_dictionary = {
        12: "December",
        1: "January",
        2: "February",
        6: "June",
        7: "July",
        8: "August",
    }
    return ds, month_dictionary


@app.cell
def _(mo):
    mo.md(r"""
    ### Warming
    """)
    return


@app.cell
def _(drop_empty_years, ds, month_dictionary, plt):
    _fig, _ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    for _month in ds["month"].values:
        _month_data = drop_empty_years(ds["nino_34"].sel(month=_month))
        _years = _month_data["year"].values
        _mean_ensemble = _month_data.mean(dim="sample", skipna=True)
        _std_ensemble = _month_data.std(dim="sample", skipna=True)

        if _month in [12, 1, 2]:
            ax1 = _ax[0]
        else:
            ax1 = _ax[1]

        ax1.plot(_years, _mean_ensemble, linewidth=2, label=month_dictionary[_month])
        ax1.fill_between(
            x=_years,
            y1=_mean_ensemble - _std_ensemble,
            y2=_mean_ensemble + _std_ensemble,
            alpha=0.2,
        )
    _ax[0].legend()
    _ax[0].grid(alpha=0.4, linestyle="--")
    _ax[1].grid(alpha=0.4, linestyle="--")
    _ax[1].legend()
    _fig.tight_layout()
    _ax[0].set_title("Future trends of TREFHT in Nino3.4 region")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Niño 3.4 index
    We define it with respect to the ensemble mean for each given month.
    """)
    return


@app.cell
def _(compute_nino34_index, ds):
    nino34_index = compute_nino34_index(diffusion_dataset=ds)
    return (nino34_index,)


@app.cell
def _(drop_empty_years, ds, month_dictionary, nino34_index, plt):
    _fig, _ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    for _month in ds["month"].values:
        _month_data = drop_empty_years(nino34_index.sel(month=_month))
        _years = _month_data["year"].values
        _mean_ensemble = _month_data.mean(dim="sample", skipna=True)
        _std_ensemble = _month_data.std(dim="sample", skipna=True)

        if _month in [12, 1, 2]:
            __ax = _ax[0]
        else:
            __ax = _ax[1]

        __ax.plot(_years, _mean_ensemble, linewidth=2, label=month_dictionary[_month])
        __ax.fill_between(
            x=_years,
            y1=_mean_ensemble - _std_ensemble,
            y2=_mean_ensemble + _std_ensemble,
            alpha=0.4,
        )
    _ax[0].legend()
    _ax[0].grid(alpha=0.4, linestyle="--")
    _ax[1].grid(alpha=0.4, linestyle="--")
    _ax[1].legend()
    _fig.tight_layout()
    _ax[0].set_title("Nino3.4 index region across ensembles")
    plt.show()
    return


@app.cell
def _(drop_empty_years, ds, month_dictionary, nino34_index, plt):
    _fig, _ax = plt.subplots()
    for _month in ds["month"].values:
        _month_data = drop_empty_years(nino34_index.sel(month=_month))
        _years = _month_data["year"].values
        _std_ensemble = _month_data.std(dim="sample", skipna=True)
        _ax.plot(_years, _std_ensemble, label=month_dictionary[_month])

    _fig.legend(loc="center left", bbox_to_anchor=(0.82, 0.84))
    _fig.tight_layout(rect=(0, 0, 0.85, 1))

    _ax.set_ylabel("Variability across ensemble")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    We concentrate on December and identify El Niño events
    """)
    return


@app.cell
def _(nino34_index, select_month):
    selected_month = 12
    nino_34_d = select_month(data_array=nino34_index, month=selected_month)
    return nino_34_d, selected_month


@app.cell
def _(nino_34_d, select_reference_samples):
    # Find reference events: max, min and neutral in a given year
    selected_year = 2030
    max_value, min_value, neutral_value = select_reference_samples(
        nino_34_d,
        year=selected_year,
    )
    return max_value, min_value, neutral_value


@app.cell
def _(max_value, min_value, neutral_value, nino_34_d, plt):
    _fig, _ax = plt.subplots()

    for _year in nino_34_d["year"].values:
        _ax.scatter(
            [ _year ] * nino_34_d.sizes["sample"],
            nino_34_d.sel(year=_year),
            marker=".",
            s=1,
            color="b",
            alpha=0.5
        )
    _ax.set_ylabel("Anomalies",fontsize=16)
    _ax.set_xlabel("Time",fontsize=16)

    _ax.scatter(min_value["year"].item(),min_value.item(),color='b',marker="*",s=56)
    _ax.scatter(max_value["year"].item(),max_value.item(),color='r',marker="*",s=56)
    _ax.scatter(neutral_value["year"].item(),neutral_value.item(),color='black',marker="*",s=56)
    plt.show()
    return


@app.cell
def _(ccrs, cfeatures):
    def plot_reference_anomaly_map(_ax, df_plot, *, title: str = ""):
        df_plot.plot(
            ax=_ax,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            center=0,
            cbar_kwargs={
                "orientation": "horizontal",
                "pad": 0.05,
                "shrink": 0.7,
                "aspect": 35,
                "label": "",
            },
        )

        _ax.coastlines(linewidth=0.8)
        _ax.add_feature(
            cfeatures.BORDERS,
            linestyle="--",
            linewidth=0.6,
            edgecolor="black",
        )
    
        lat_min , lat_max = df_plot["lat"].min() , df_plot["lat"].max()
        _ax.set_extent([0, 360,lat_max, lat_min], crs=ccrs.PlateCarree())
        _ax.set_title(title)

        _gl = _ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.6,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        _gl.top_labels = False
        _gl.right_labels = False
        _gl.xlabel_style = {"size": 10}
        _gl.ylabel_style = {"size": 10}

    return (plot_reference_anomaly_map,)


@app.cell
def _(
    ccrs,
    ds,
    input_data_folder_path,
    max_value,
    min_value,
    neutral_value,
    plot_reference_anomaly_map,
    plt,
    reconstruct_sample_anomaly,
    selected_month,
):
    _fig, _ax = plt.subplots(
        nrows= 3,
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )

    titles = ["El Niño", "La Niña", "El Ñeutral"]
    for index, value in enumerate( [max_value,min_value,neutral_value] ):
        df_anomaly = reconstruct_sample_anomaly(
            diffusion_dataset=ds,
            input_data_dir=input_data_folder_path,
            month=selected_month,
            year=value["year"].item(),
            sample=value["sample"].item(),
            variable="TREFHT"
        )
        df_plot = df_anomaly.sel(lat=slice(-45, 45))
        plot_reference_anomaly_map(_ax[index], df_plot, title=titles[index])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
