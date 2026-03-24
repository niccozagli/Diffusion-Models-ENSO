from __future__ import annotations

from diffusion_models_enso.utils import drop_empty_years

DJF_MONTHS = ("December", "January", "February")


def plot_monthly_mean_with_spread(
    data_array,
    *,
    plt,
    title: str,
    ylabel: str = "",
    alpha: float = 0.2,
):
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    for month in data_array["month"].values:
        month_data = drop_empty_years(data_array.sel(month=month))
        years = month_data["year"].values
        mean_ensemble = month_data.mean(dim="sample", skipna=True)
        std_ensemble = month_data.std(dim="sample", skipna=True)
        axis = axes[0] if month in DJF_MONTHS else axes[1]

        axis.plot(years, mean_ensemble, linewidth=2, label=month)
        axis.fill_between(
            x=years,
            y1=mean_ensemble - std_ensemble,
            y2=mean_ensemble + std_ensemble,
            alpha=alpha,
        )

    axes[0].legend()
    axes[0].grid(alpha=0.4, linestyle="--")
    axes[1].grid(alpha=0.4, linestyle="--")
    axes[1].legend()
    fig.suptitle(title)
    if ylabel:
        axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig, axes


def plot_monthly_ensemble_variability(
    data_array,
    *,
    plt,
    ylabel: str = "Variability across ensemble",
):
    fig, axis = plt.subplots()

    for month in data_array["month"].values:
        month_data = drop_empty_years(data_array.sel(month=month))
        years = month_data["year"].values
        std_ensemble = month_data.std(dim="sample", skipna=True)
        axis.plot(years, std_ensemble, label=month)

    fig.legend(loc="center left", bbox_to_anchor=(0.82, 0.84))
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    axis.set_ylabel(ylabel)
    return fig, axis


def plot_reference_anomaly_map(
    ax,
    data_array,
    *,
    ccrs,
    cfeatures,
    title: str = "",
    add_colorbar: bool = False,
    cbar_kwargs: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    plot_kwargs = {
        "ax": ax,
        "transform": ccrs.PlateCarree(),
        "cmap": "RdBu_r",
        "center": 0,
        "add_colorbar": add_colorbar,
        "vmin": vmin,
        "vmax": vmax,
    }
    if add_colorbar:
        plot_kwargs["cbar_kwargs"] = cbar_kwargs or {
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
            "aspect": 35,
            "label": "",
        }

    mappable = data_array.plot(**plot_kwargs)

    ax.coastlines(linewidth=0.8)
    ax.add_feature(
        cfeatures.BORDERS,
        linestyle="--",
        linewidth=0.6,
        edgecolor="black",
    )

    lat_min = data_array["lat"].min().item()
    lat_max = data_array["lat"].max().item()
    ax.set_extent([0, 360, lat_max, lat_min], crs=ccrs.PlateCarree())
    ax.set_title(title)

    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 10}
    gridlines.ylabel_style = {"size": 10}

    return mappable


def plot_yearly_composite_grid(
    composites,
    event_counts,
    *,
    plt,
    ccrs,
    cfeatures,
    plot_reference_anomaly_map_fn,
    title: str,
    lat_slice: tuple[float, float] | None = None,
    symmetric: bool = True,
    figsize: tuple[float, float] = (10, 7),
):
    composite_years = composites["year"].values.tolist()

    if lat_slice is not None:
        plot_composites = composites.sel(lat=slice(*lat_slice))
    else:
        plot_composites = composites

    if symmetric:
        vmax = plot_composites.max(skipna=True).item()
        vmin = -vmax
    else:
        vmin = plot_composites.min(skipna=True).item()
        vmax = plot_composites.max(skipna=True).item()

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, year, count in zip(
        axes,
        composite_years,
        event_counts.values.tolist(),
        strict=True,
    ):
        mappable = plot_reference_anomaly_map_fn(
            ax,
            plot_composites.sel(year=year),
            ccrs=ccrs,
            cfeatures=cfeatures,
            title=f"Year {year} (n={count})",
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle(title, fontsize=14)
    fig.colorbar(
        mappable,
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.03,
    )
    return fig, axes
