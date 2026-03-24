import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import cartopy.crs as ccrs
    import cartopy.feature as cfeatures
    import marimo as mo
    import xarray as xr
    from matplotlib import pyplot as plt
    from plotting_utils import (
        plot_monthly_ensemble_variability,
        plot_monthly_mean_with_spread,
        plot_reference_anomaly_map,
        plot_yearly_composite_grid,
    )

    from diffusion_models_enso.analysis import prepare_selected_event_anomaly
    from diffusion_models_enso.utils import find_repo_root

    return (
        Path,
        ccrs,
        cfeatures,
        find_repo_root,
        mo,
        plot_monthly_ensemble_variability,
        plot_monthly_mean_with_spread,
        plot_reference_anomaly_map,
        plot_yearly_composite_grid,
        plt,
        prepare_selected_event_anomaly,
        xr,
    )


@app.cell
def _(Path, find_repo_root):
    repo_folder_path = find_repo_root(Path(__file__))
    data_folder_path = repo_folder_path / "data"
    return (data_folder_path,)


@app.cell
def _(data_folder_path, xr):
    ds = xr.open_dataset(filename_or_obj=data_folder_path / "monthly_diagnostics.nc")
    return (ds,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Diagnostics plots
    """)
    return


@app.cell
def _(ds, plot_monthly_mean_with_spread, plt):
    _fig, _ax = plot_monthly_mean_with_spread(
        ds["nino34_trefht"],
        plt=plt,
        title="Future trends of TREFHT in Nino3.4 region",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Niño 3.4 index
    It is defined with respect to the ensemble mean for each given month.
    """)
    return


@app.cell
def _(ds, plot_monthly_mean_with_spread, plt):
    _nino34_index = ds["nino34_index"]
    _fig, _ax = plot_monthly_mean_with_spread(
        data_array=_nino34_index,
        plt=plt,
        title="Nino3.4 index region across ensembles",
        alpha=0.4,
    )
    plt.show()
    return


@app.cell
def _(ds, plot_monthly_ensemble_variability, plt):
    _nino34_index = ds["nino34_index"]
    _fig, _ax = plot_monthly_ensemble_variability(
        data_array=ds["nino34_index"],
        plt=plt,
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### DJF scatter
    """)
    return


@app.cell
def _(ds, plt):
    _nino34_index = ds["nino34_index"]
    _djf_colors = {
        "December": "tab:blue",
        "January": "tab:orange",
        "February": "tab:green",
    }
    _djf_index = _nino34_index.sel(month=list(_djf_colors))

    _fig, _ax = plt.subplots()

    for _month, _color in _djf_colors.items():
        _month_data = _djf_index.sel(month=_month)
        _first_year = _month_data["year"].values[0]

        for _year in _month_data["year"].values:
            _year_values = _month_data.sel(year=_year).dropna(dim="sample")
            if _year_values.sizes["sample"] == 0:
                continue
            _ax.scatter(
                [_year] * _year_values.sizes["sample"],
                _year_values.values,
                marker=".",
                s=8,
                color=_color,
                alpha=0.5,
                label=_month if _year == _first_year else None,
            )

    _ax.set_ylabel("Anomalies", fontsize=16)
    _ax.set_xlabel("Time", fontsize=16)
    _ax.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ENSO events
    """)
    return


@app.cell
def _(data_folder_path, prepare_selected_event_anomaly, xr):
    enso = xr.open_dataset(filename_or_obj=data_folder_path / "enso_selection.nc")

    selected_enso_event = enso.where(
        cond=(enso["month"] == "December") & (enso["year"] == 2015),
        drop=True,
    )

    strongest_nino = selected_enso_event.isel(
        event=selected_enso_event["nino34_index"].argmax(dim="event").item()
    )

    strongest_nina = selected_enso_event.isel(
        event=selected_enso_event["nino34_index"].argmin(dim="event").item()
    )

    anomaly_nino = prepare_selected_event_anomaly(selected_event=strongest_nino)
    anomaly_nina = prepare_selected_event_anomaly(selected_event=strongest_nina)
    return anomaly_nina, anomaly_nino


@app.cell
def _(
    anomaly_nina,
    anomaly_nino,
    ccrs,
    cfeatures,
    plot_reference_anomaly_map,
    plt,
):
    _nino_plot = anomaly_nino.sel(lat=slice(-20, 20))
    _nina_plot = anomaly_nina.sel(lat=slice(-20, 20))
    _vmax = _nino_plot.max().item()
    _vmin = -_vmax

    _fig, _ax = plt.subplots(
        nrows=2,
        figsize=(8, 4),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )

    _mappable = plot_reference_anomaly_map(
        _ax[0],
        _nino_plot,
        ccrs=ccrs,
        cfeatures=cfeatures,
        title=f"Strongest El Nino anomaly in {anomaly_nino.attrs['month']} {anomaly_nino.attrs['year']}",
        add_colorbar=False,
        vmin=_vmin,
        vmax=_vmax,
    )
    plot_reference_anomaly_map(
        _ax[1],
        _nina_plot,
        ccrs=ccrs,
        cfeatures=cfeatures,
        title=f"Strongest La Nina anomaly in {anomaly_nina.attrs['month']} {anomaly_nina.attrs['year']}",
        add_colorbar=False,
        vmin=_vmin,
        vmax=_vmax,
    )
    _fig.colorbar(
        _mappable,
        ax=_ax,
        orientation="horizontal",
        shrink=0.5,
        pad=0.08,
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ENSO composites
    """)
    return


@app.cell
def _(data_folder_path, xr):
    enso_composites = xr.open_dataset(filename_or_obj=data_folder_path / "enso_composites.nc")
    selected_months = enso_composites.attrs.get("selection_months", "").split(',')
    return enso_composites, selected_months


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["trefht_composite"].sel(event_type="nino")
    _event_counts = enso_composites["n_events"].sel(event_type="nino")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nino TREFHT composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["trefht_composite"].sel(event_type="nina")
    _event_counts = enso_composites["n_events"].sel(event_type="nina")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nina TREFHT composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["prect_composite"].sel(event_type="nino")
    _event_counts = enso_composites["n_events"].sel(event_type="nino")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nino PRECT composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["prect_composite"].sel(event_type="nina")
    _event_counts = enso_composites["n_events"].sel(event_type="nina")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nina PRECT composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["ps_composite"].sel(event_type="nino")
    _event_counts = enso_composites["n_events"].sel(event_type="nino")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nino PS composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


@app.cell
def _(
    ccrs,
    cfeatures,
    enso_composites,
    plot_reference_anomaly_map,
    plot_yearly_composite_grid,
    plt,
    selected_months,
):
    _composites = enso_composites["ps_composite"].sel(event_type="nina")
    _event_counts = enso_composites["n_events"].sel(event_type="nina")
    _fig, _axes = plot_yearly_composite_grid(
        _composites,
        _event_counts,
        plt=plt,
        ccrs=ccrs,
        cfeatures=cfeatures,
        plot_reference_anomaly_map_fn=plot_reference_anomaly_map,
        title=f"Nina PS composites for {', '.join(selected_months)}",
        lat_slice=(-45, 45),
        symmetric=True,
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
