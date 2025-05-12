# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# def plot_ENSO_NA(slopes_diffy):
#     levels = list(np.arange(-2.7, 2.75, 0.05))
#     cmap = plt.get_cmap("RdBu_r")
#     norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

#     # Define fewer ticks for the colorbar
#     fewer_ticks = [
#         -7e-9,
#         -5e-9,
#         -3e-9,
#         -1e-9,
#         1e-9,
#         3e-9,
#         5e-9,
#         7e-9,
#     ]  # Select fewer representative levels

#     # Plotting
#     fig, ax = plt.subplots(
#         1,
#         1,
#         figsize=(10, 5),
#         subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
#     )

#     # Set extent to zoom in on a particular region
#     ax.set_extent(
#         [120, 300, -35, 35], crs=ccrs.PlateCarree()
#     )  # [min_lon, max_lon, min_lat, max_lat]

#     # Color-filled plot for TREFHT with discrete levels
#     prect_plot = slopes_diffy["TREFHT"].plot(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#         norm=norm,
#         add_colorbar=True,
#         cbar_kwargs={
#             "boundaries": levels,
#             "ticks": fewer_ticks,
#             "shrink": 0.7,
#         },  # Shrink and reduce the number of ticks
#     )
#     plt.title("Regional Zoom: TREFHT (color) and PS (contours)")

#     # Adding contour lines for PS
#     contour = ax.contour(
#         slopes_diffy["PS"]["lon"],
#         slopes_diffy["PS"]["lat"],
#         slopes_diffy["PS"],
#         levels=10,
#         colors="black",
#         transform=ccrs.PlateCarree(),
#     )
#     ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f")

#     # Adding coastlines and formatting
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=":")
#     ax.set_xticks(
#         [120, 150, 180, 210, 240], crs=ccrs.PlateCarree()
#     )  # Adjusted for the zoomed region
#     ax.set_yticks(
#         [10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree()
#     )  # Adjusted for the zoomed region
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}°"))
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}°"))

#     plt.tight_layout()
#     plt.show()


# def plot_ENSO_NA_diff(slopes_diffy1, slopes_diffy2):
#     levels = list(np.arange(-0.7, 0.75, 0.05) * 1e-8)
#     cmap = plt.get_cmap("RdBu_r")
#     norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

#     # Define fewer ticks for the colorbar
#     fewer_ticks = [
#         -7e-9,
#         -5e-9,
#         -3e-9,
#         -1e-9,
#         1e-9,
#         3e-9,
#         5e-9,
#         7e-9,
#     ]  # Select fewer representative levels

#     # Plotting
#     fig, ax = plt.subplots(
#         1,
#         1,
#         figsize=(10, 5),
#         subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
#     )

#     # Set extent to zoom in on a particular region
#     # ax.set_extent([120, 300, -10, 80], crs=ccrs.PlateCarree())  # [min_lon, max_lon, min_lat, max_lat]

#     # Color-filled plot for TREFHT with discrete levels
#     prect_plot = (slopes_diffy1["PRECT"] - slopes_diffy2["PRECT"]).plot(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#         norm=norm,
#         add_colorbar=True,
#         cbar_kwargs={
#             "boundaries": levels,
#             "ticks": fewer_ticks,
#             "shrink": 0.7,
#         },  # Shrink and reduce the number of ticks
#     )
#     plt.title("Regional Zoom: PRECT (color) and PS (contours)")

#     # Adding contour lines for PS
#     contour = ax.contour(
#         slopes_diffy1["PS"]["lon"],
#         slopes_diffy1["PS"]["lat"],
#         slopes_diffy1["PS"] - slopes_diffy2["PS"],
#         levels=10,
#         colors="black",
#         transform=ccrs.PlateCarree(),
#     )
#     ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f")

#     # Adding coastlines and formatting
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=":")
#     ax.set_xticks(
#         [120, 150, 180, 210, 240], crs=ccrs.PlateCarree()
#     )  # Adjusted for the zoomed region
#     ax.set_yticks(
#         [10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree()
#     )  # Adjusted for the zoomed region
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}°"))
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}°"))

#     plt.tight_layout()
#     plt.show()


# def compute_regression_slopes(
#     DS, vardo="TREFHT", region_lat=(-5, 5), region_lon=(190, 240)
# ):
#     """
#     Compute the regression slopes of each variable (PS, PRECT, TREFHT) with respect to the mean of TREFHT
#     over a specified region.

#     Parameters:
#     DS (xarray.Dataset): The dataset containing the variables.
#     region_lat (tuple): Latitude bounds for selecting the region (default: (-5, 5)).
#     region_lon (tuple): Longitude bounds for selecting the region (default: (190, 240)).

#     Returns:
#     dict: A dictionary containing the slope datasets for each variable (PS, PRECT, TREFHT).
#     """

#     # Step 1: Select the mean of TREFHT over the specified region
#     trefht_mean = (
#         DS[vardo]
#         .sel(lat=slice(*region_lat), lon=slice(*region_lon))
#         .mean(dim=["lat", "lon"])
#     )

#     # Step 2: Demean the independent variable (TREFHT)
#     trefht_mean_demeaned = trefht_mean - trefht_mean.mean(dim="samples")

#     # Step 3: Loop over each variable and compute the slope at each (lat, lon) point
#     regression_slopes = {}
#     for var_name in ["PS", "PRECT", "TREFHT"]:
#         # Demean the dependent variable at each (lat, lon) point
#         y = DS[var_name]
#         y_demeaned = y - y.mean(dim="samples")

#         # Compute the slope (cov(y, x) / var(x)) at each (lat, lon) point
#         slope = (y_demeaned * trefht_mean_demeaned).mean(dim="samples") / (
#             trefht_mean_demeaned**2
#         ).mean(dim="samples")

#         # Store the slope result for this variable
#         regression_slopes[var_name] = slope

#     return regression_slopes


def plot_ENZO(X, Y, ax, title):
    colors = ["g", "m", "b"]
    labels = ["Dec", "Jan", "Feb"]
    for i in range(len(X)):
        x_means, x_var = X[i], Y[i]  # X_means[i] , X_var[i]
        ax.fill_between(
            range(0, len(x_means)),
            x_means - x_var,
            x_means + x_var,
            alpha=0.12,
            color=colors[i],
        )
        ax.plot(x_means, color=colors[i], label=labels[i])
    ax.legend(fontsize=15)
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=18)
