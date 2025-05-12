import numpy as np


def get_statistics(paths):
    X_means = []
    X_all = []
    X_var = []

    for path in paths:
        x = np.load(path, allow_pickle=True).item()
        x_means = []
        x_all = dict()
        x_var = []
        for key, value in x.items():
            media = np.mean(value)
            std = np.std(value)
            x_means.append(media)
            x_var.append(std)
            x_all[key] = value

        x_means = np.array(x_means)
        x_var = np.array(x_var)

        X_means.append(x_means)
        X_all.append(x_all)
        X_var.append(x_var)

    return X_means, X_var, X_all


def evaluate_anomalies(months, Dataset_Nino, Dataset_Global):
    ANOMALIES = []
    before_removing_ensemble_mean = []
    for ii, month in enumerate(months):
        data_month = Dataset_Nino[ii]
        means = Dataset_Global[ii]

        anomalies = dict()
        for key, value in data_month.items():
            anomalies[key] = data_month[key] - means[key]

        mean_anomaly = [np.mean(x) for x in anomalies.values()]
        for key, value in data_month.items():
            anomalies[key] = anomalies[key] - mean_anomaly[key]

        ANOMALIES.append(anomalies)
        before_removing_ensemble_mean.append(mean_anomaly)
    return ANOMALIES, before_removing_ensemble_mean
