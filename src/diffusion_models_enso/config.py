# from pydantic_settings import BaseSettings


class DataRetrievinSettings:  # (BaseSettings):
    ENSO_lat_min: float = -5
    ENSO_lat_max: float = +5
    ENSO_long_min: float = 190
    ENSO_long_max: float = 240

    diffusion_data_path: str = (
        "//users_home/training/data/Group_01/Ensembles/samples_governance_indexes_3944_month"
    )
    lens_data_path: str = "//users_home/training/data/Group_01/LENS"
