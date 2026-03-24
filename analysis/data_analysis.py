import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import xarray as xr
    from diffusion_models_enso.analysis import (
        build_event_anomaly_matrix,
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
    from sklearn.decomposition import PCA
    import numpy as np

    return (
        PCA,
        Path,
        build_event_anomaly_matrix,
        ccrs,
        cfeatures,
        compute_nino34_index,
        drop_empty_years,
        find_repo_root,
        mo,
        np,
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
    ds = xr.open_dataset(filename_or_obj=data_folder_path / "monthly_diagnostics.nc")
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
        _month_data = drop_empty_years(ds["nino34_index"].sel(month=_month))
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
        df_plot = df_anomaly.sel(lat=slice(-30, 30))
        plot_reference_anomaly_map(_ax[index], df_plot, title=titles[index])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ENSO analysis

    For the time being, we fix December and we define as ENSO events all fields where the Nino3.4 index is above a given threshold, obtained as a quantile.
    """)
    return


@app.cell
def _(nino_34_d):
    def get_quantile_threshold(ds,alpha):
        ds = ds.stack(event=("year","sample")).dropna("event")
        threshold = ds.quantile(alpha,dim="event").item()
        return threshold


    threshold_high = get_quantile_threshold(ds=nino_34_d,alpha=0.80)
    el_nino_dec = (
        nino_34_d
        .stack(event=("year", "sample"))
        .where(lambda x: x > threshold_high, drop=True)
    )

    threshold_low = get_quantile_threshold(ds=nino_34_d,alpha=0.20)
    la_nina_dec = (
        nino_34_d
        .stack(event=("year","sample"))
        .where(lambda x : x < threshold_low,drop=True)
    )
    return el_nino_dec, threshold_high, threshold_low


@app.cell
def _(nino_34_d, plt, threshold_high, threshold_low):
    _fig, _ax = plt.subplots()


    for _year in nino_34_d["year"].values:
        _year_values = nino_34_d.sel(year=_year)
        _el_nino = _year_values.where(_year_values > threshold_high, drop=True)
        _la_nina = _year_values.where(_year_values < threshold_low, drop=True)
        _neutral = _year_values.where(
            (_year_values >= threshold_low) & (_year_values <= threshold_high),
            drop=True,
        )

        _ax.scatter(
            [_year] * _neutral.sizes["sample"],
            _neutral,
            marker=".",
            s=1,
            color="0.7",
            alpha=0.4,
        )
        _ax.scatter(
            [_year] * _la_nina.sizes["sample"],
            _la_nina,
            marker="*",
            s=1,
            color="tab:blue",
            alpha=1,
        )
        _ax.scatter(
            [_year] * _el_nino.sizes["sample"],
            _el_nino,
            marker="*",
            s=1,
            color="tab:red",
            alpha=1,
        )
    _ax.set_ylabel("Anomalies", fontsize=16)
    _ax.set_xlabel("Time", fontsize=16)


    plt.show()
    return


@app.cell
def _(build_event_anomaly_matrix, ds, el_nino_dec, input_data_folder_path):
    event_anomaly_matrix = build_event_anomaly_matrix(
        diffusion_dataset=ds,
        events=el_nino_dec,
        input_data_dir=input_data_folder_path,
        variable="TREFHT",
        lat_bounds=(-20, 20),
        lon_bounds=(120, 280),
        num_workers=4,
    )
    return (event_anomaly_matrix,)


@app.cell
def _(ccrs, event_anomaly_matrix, plt):
    # Plot the mean ENSO field across all the samples (across years)
    mean_anomaly = event_anomaly_matrix.mean(dim="event").unstack(dim="feature")
    _fig, _ax = plt.subplots(
        figsize=(8,5),
        subplot_kw={"projection":ccrs.PlateCarree(central_longitude=180)})

    mean_anomaly.plot(
        ax=_ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
                "orientation": "horizontal",
                "pad": 0.05,
                "shrink": 0.7,
                "aspect": 35,
                "label": "TREFHT",
            },
        add_labels=False 
    ) # type:ignore
    _ax.coastlines()
    _ax.set_title("Mean ENSO anomaly across all ENSO events")
    plt.show()
    return


@app.cell
def _(PCA, event_anomaly_matrix):
    # Performing PCA
    X = event_anomaly_matrix.values
    X_mean = event_anomaly_matrix.mean(dim="event").values
    X_centred = X - X_mean

    pca = PCA()
    pca.fit(X_centred)
    return X_centred, pca


@app.cell
def _(ccrs, event_anomaly_matrix, pca, plt, xr):
    EOFS = pca.components_

    n_row, n_col = 3, 2
    _fig, _ax = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        figsize=(10, 6.5),
        constrained_layout=True,
    )

    template = event_anomaly_matrix.isel(event=0).unstack("feature")
    lat, lon = template["lat"].values, template["lon"].values

    mappable = None

    for _index in range(n_row * n_col):
        ax = _ax.ravel()[_index]
        row = _index // n_col
        col = _index % n_col

        eof = EOFS[_index, :].reshape((len(lat), len(lon)))
        eof_xr = xr.DataArray(
            data=eof,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        mappable = eof_xr.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            add_colorbar=False,
            add_labels=False,
        ) #type:ignore
        ax.coastlines()
        ax.set_title(
            f"EOF {_index + 1}: var explained {pca.explained_variance_ratio_[_index] * 100:.1f}%"
        )

        _gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        _gl.top_labels = False
        _gl.right_labels = False
        _gl.left_labels = col == 0
        _gl.bottom_labels = row == n_row - 1
        _gl.xlabel_style = {"size": 8}
        _gl.ylabel_style = {"size": 8}

    _fig.colorbar(
        mappable,
        ax=_ax.ravel(),
        orientation="horizontal",
        shrink=0.7,
        pad=0.06,
    )

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Truncation
    """)
    return


@app.cell
def _(np, pca, plt):
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    threshold = 0.95
    r = np.argmax(cum_var > threshold) 

    _fig, _ax = plt.subplots()
    _ax.semilogx(cum_var)
    _ax.set_ylabel("Cumulative Variance")
    _ax.hlines(y=threshold,xmin=0,xmax=len(cum_var),linestyles='--',color='gray')
    return (r,)


@app.cell
def _(X_centred, event_anomaly_matrix, pca, r):
    Z = pca.transform(X=X_centred)
    EOFS_r = pca.components_[:r+1,:]
    Zr = Z[:,:r+1]

    mean_anomaly_raveled = event_anomaly_matrix.mean(dim="event").values
    X_reconstructed = Zr @ EOFS_r + mean_anomaly_raveled

    event_anomaly_reconstructed = event_anomaly_matrix.copy(
        data=X_reconstructed
    )
    event_anomaly_reconstructed.name = "reconstructed_anomaly"
    event_anomaly_reconstructed.attrs["truncation"] = r+1
    return Zr, event_anomaly_reconstructed


@app.cell
def _(event_anomaly_matrix, event_anomaly_reconstructed, np, plt):
    residual = (event_anomaly_matrix - event_anomaly_reconstructed)**2
    rmse = np.sqrt( residual.mean(dim="event") )
    _fig, _ax = plt.subplots()
    _ax.hist(x=rmse,bins=50)
    return


@app.cell
def _(event_anomaly_matrix):
    event_anomaly_matrix
    return


@app.cell
def _(ccrs, event_anomaly_matrix, event_anomaly_reconstructed, np, plt):
    sample_event = 0

    x = event_anomaly_matrix.isel(event=sample_event).unstack(dim="feature")
    x_reconstructed = event_anomaly_reconstructed.isel(event=sample_event).unstack(dim="feature")


    vmax = np.max(np.abs([x.values, x_reconstructed.values]))
    vmin = -vmax

    _fig, _ax = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )

    _mappable = x.plot(
        ax=_ax[0],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        add_labels=False,
    )
    _ax[0].coastlines()
    _ax[0].set_title("Real Anomaly")

    x_reconstructed.plot(
        ax=_ax[1],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        add_labels=False,
    )
    _ax[1].coastlines()
    _ax[1].set_title("Reconstructed Anomaly")

    _fig.colorbar(
        _mappable,
        ax=_ax,
        orientation="horizontal",
        shrink=0.7,
        pad=0.06,
    )

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cluster Analysis
    First option: we are not normalising the PCA components (so there is a hierarchy, mode 1 counts more than mode2 etc...)
    """)
    return


@app.cell
def _(Zr, event_anomaly_matrix, xr):
    from sklearn.cluster import KMeans

    k = 3
    kmeans = KMeans(n_clusters=k, n_init="auto")
    labels = kmeans.fit_predict(X=Zr)

    cluster_labels = xr.DataArray(
        labels,
        dims=("event",),
        coords={"event": event_anomaly_matrix["event"]},
        name="cluster",
    )
    return KMeans, cluster_labels, k


@app.cell
def _(Zr, cluster_labels, k, np, plt):
    cmap = plt.get_cmap("tab10", k)

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _scatter = _ax.scatter(
        Zr[:, 0],
        Zr[:, 1],
        c=cluster_labels.values,
        cmap=cmap,
        s=10,
        vmin=-0.5,
        vmax=k - 0.5,
    )

    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.set_title("KMeans clusters in reduced space")

    _cbar = _fig.colorbar(_scatter, ax=_ax, ticks=np.arange(k))
    _cbar.set_label("Cluster")
    _cbar.set_ticklabels([str(i) for i in range(k)])

    plt.show()
    return


@app.cell
def _(ccrs, cluster_labels, event_anomaly_matrix, k, plt):
    _fig, _ax = plt.subplots(
        nrows=k,
        figsize=(8, 3 * k),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )

    if k == 1:
        _ax = [_ax]

    _mappable = None

    for _cluster in range(k):
        mask = cluster_labels.values == _cluster

        cluster_mean = (
            event_anomaly_matrix
            .isel(event=mask)
            .mean(dim="event")
            .unstack("feature")
        )

        _mappable = cluster_mean.plot(
            ax=_ax[_cluster],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            add_colorbar=False,
            add_labels=False,
        )
        _ax[_cluster].coastlines()
        _ax[_cluster].set_title(f"Cluster {_cluster} mean anomaly (n={mask.sum()})")

    _fig.colorbar(
        _mappable,
        ax=_ax,
        orientation="horizontal",
        shrink=0.7,
        pad=0.06,
    )

    plt.show()
    return


@app.cell
def _(KMeans, Zr, ccrs, event_anomaly_matrix, k, np, plt, xr):
    from sklearn.preprocessing import StandardScaler

    scaler_zr = StandardScaler()
    Zr_scaled = scaler_zr.fit_transform(Zr)

    kmeans_scaled = KMeans(n_clusters=k, n_init="auto")
    labels_scaled = kmeans_scaled.fit_predict(X=Zr_scaled)

    cluster_labels_scaled = xr.DataArray(
        labels_scaled,
        dims=("event",),
        coords={"event": event_anomaly_matrix["event"]},
        name="cluster_scaled",
    )

    cmap_scaled = plt.get_cmap("tab10", k)

    _fig_scaled, _ax_scaled = plt.subplots(figsize=(6, 5))
    _scatter_scaled = _ax_scaled.scatter(
        Zr_scaled[:, 0],
        Zr_scaled[:, 1],
        c=cluster_labels_scaled.values,
        cmap=cmap_scaled,
        s=10,
        vmin=-0.5,
        vmax=k - 0.5,
    )

    _ax_scaled.set_xlabel("Scaled PC1")
    _ax_scaled.set_ylabel("Scaled PC2")
    _ax_scaled.set_title("KMeans clusters in scaled reduced space")

    _cbar_scaled = _fig_scaled.colorbar(_scatter_scaled, ax=_ax_scaled, ticks=np.arange(k))
    _cbar_scaled.set_label("Cluster")
    _cbar_scaled.set_ticklabels([str(i) for i in range(k)])

    plt.show()

    _fig_maps_scaled, _ax_maps_scaled = plt.subplots(
        nrows=k,
        figsize=(8, 3 * k),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )

    if k == 1:
        _ax_maps_scaled = [_ax_maps_scaled]

    _mappable_scaled = None

    for _cluster_scaled in range(k):
        mask_scaled = cluster_labels_scaled.values == _cluster_scaled

        cluster_mean_scaled = (
            event_anomaly_matrix
            .isel(event=mask_scaled)
            .mean(dim="event")
            .unstack("feature")
        )

        _mappable_scaled = cluster_mean_scaled.plot(
            ax=_ax_maps_scaled[_cluster_scaled],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            add_colorbar=False,
            add_labels=False,
        )
        _ax_maps_scaled[_cluster_scaled].coastlines()
        _ax_maps_scaled[_cluster_scaled].set_title(
            f"Scaled cluster {_cluster_scaled} mean anomaly (n={mask_scaled.sum()})"
        )

    _fig_maps_scaled.colorbar(
        _mappable_scaled,
        ax=_ax_maps_scaled,
        orientation="horizontal",
        shrink=0.7,
        pad=0.06,
    )

    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
