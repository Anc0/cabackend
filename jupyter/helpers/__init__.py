import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import random
import warnings

from collections import Counter
from datetime import datetime, timedelta
from fastdtw import fastdtw
from IPython.display import clear_output
from math import sqrt, log10
from numpy import mean, std, array
from numpy.fft import fft, fftfreq, ifftshift
from pandas import DataFrame, read_csv, concat
from plotly.express import scatter, line, scatter_3d, bar, line_3d
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from seances.models import Seance
from sensors.models import Sensor, SensorRecord


def load_data(seance_id, sens):
    seance = Seance.objects.get(id=seance_id)
    if sens == "accelerometer":
        sensor_ids = [60, 61, 62]

        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("id")
        return (
            SensorRecord.objects.filter(seance=seance, sensor=sensors[0]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[1]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[2]).order_by(
                "timestamp"
            ),
        )
    elif sens == "gyroscope":
        sensor_ids = [63, 64, 65]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("id")
        return (
            SensorRecord.objects.filter(seance=seance, sensor=sensors[0]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[1]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[2]).order_by(
                "timestamp"
            ),
        )
    elif sens == "force":
        sensor_ids = [54, 55, 76, 77]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("topic")
        return (
            SensorRecord.objects.filter(
                seance=seance, sensor=sensors[0], value__gte=50
            ).order_by("timestamp"),
            SensorRecord.objects.filter(
                seance=seance, sensor=sensors[1], value__gte=50
            ).order_by("timestamp"),
            SensorRecord.objects.filter(
                seance=seance, sensor=sensors[2], value__gte=50
            ).order_by("timestamp"),
            SensorRecord.objects.filter(
                seance=seance, sensor=sensors[3], value__gte=50
            ).order_by("timestamp"),
        )
    elif sens == "cpu":
        sensor_ids = [78, 79, 80, 81]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("topic")
        return (
            SensorRecord.objects.filter(seance=seance, sensor=sensors[0]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[1]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[2]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[3]).order_by(
                "timestamp"
            ),
        )
    elif sens == "ram":
        sensor_ids = [82]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("topic")
        return SensorRecord.objects.filter(seance=seance, sensor=sensors[0]).order_by(
            "timestamp"
        )
    elif sens == "net":
        sensor_ids = [83, 84]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("id")
        return (
            SensorRecord.objects.filter(seance=seance, sensor=sensors[0]).order_by(
                "timestamp"
            ),
            SensorRecord.objects.filter(seance=seance, sensor=sensors[1]).order_by(
                "timestamp"
            ),
        )
    elif sens == "pir":
        sensor_ids = [58, 59, 66, 67, 68, 69]
        sensors = Sensor.objects.filter(id__in=sensor_ids).order_by("id")
        return SensorRecord.objects.filter(seance=seance, sensor__in=sensors).order_by(
            "timestamp"
        )
    else:
        raise ValueError("Invalid sensor string.")


def process_signal(records):
    """
    Take Django query and do basic signal processing.
    """
    values = [x.value for x in records]
    times = [x.timestamp for x in records]
    m = mean(values)
    s = std(values)
    norm = [(x - m) / s for x in values]

    return values, times, norm, m, s


def join_accelerometer_signals(x, y, z):
    """
    Join accelerometer signals, based simply on concurrence.
    We can do this, as only one controller sends data in loop for all axis.
    """
    result = []
    n = min(len(x), len(y), len(z))
    for a, b, c in zip(x[:n], y[:n], z[:n]):
        result.append(sqrt(a ** 2 + b ** 2 + c ** 2))
    return result, mean(result), std(result)


def mean_crossing_rate(signal, m):
    """
    Calculate mean crossing rate from signal.
    Rate of mean crossings vs. the signal length.
    """
    try:
        prev = signal[0]
    except IndexError:
        return 0
    crosses = 0
    length = len(signal) - 1

    for curr in signal[1:]:
        if prev <= m < curr or prev > m >= curr:
            crosses += 1
        prev = curr
    if length < 1:
        return 0
    return crosses / length


def mean_acceleration_intensity(signal):
    """
    Mean derivative of a signal.
    """
    try:
        prev = signal[0]
    except IndexError:
        return 0
    length = len(signal) - 1
    derv = []

    for curr in signal[1:]:
        derv.append(abs(curr - prev))
        prev = curr

    return mean(derv)


def join_cpu_signals(a, b, c, d):
    """
    Similar to accelerometer one.
    """
    result = []
    n = min(len(a), len(b), len(c), len(d))
    for w, x, y, z in zip(a[:n], b[:n], c[:n], d[:n]):
        result.append(sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2))
    return result, mean(result), std(result)


def get_cpu_stats(val):
    if not val:
        return 0, 0, 0
    return min(val), max(val), mean_crossing_rate(val, mean(val))


def find_ram_jump(signal):
    if not signal:
        return [], {}
    derivative = []
    prev = signal[0]
    for curr in signal[1:]:
        derivative.append(abs(curr - prev))
        prev = curr
    peaks, _ = find_peaks(derivative, threshold=0.25)

    p = {"position": [], "magnitude": []}
    for x in peaks:
        p["position"].append(x)
        p["magnitude"].append(derivative[x])
    return derivative, p


def get_mem_stats(val, peaks, derivatives):
    # Calculate average inter jump interval
    intervals = []
    if peaks and peaks["position"]:
        prev = peaks["position"][0]
        for curr in peaks["position"][1:]:
            intervals.append(curr - prev)
            prev = curr
    if val:
        avg_load = round(mean(val), 2)
        min_load = min(val)
        max_load = max(val)
    else:
        avg_load = 0
        min_load = 0
        max_load = 0
    if peaks:
        jump_count = len(peaks["position"])
        if derivatives:
            jump_rate = round(len(peaks["position"]) / len(derivatives), 2)
        else:
            jump_rate = 0
        avg_jump_value = round(mean(peaks["magnitude"]), 2)
        avg_inter_jump_interval = round(mean(intervals), 2)
    else:
        jump_count = 0
        jump_rate = 0
        avg_jump_value = 0
        avg_inter_jump_interval = 0
    return (
        avg_load,
        min_load,
        max_load,
        jump_count,
        jump_rate,
        avg_jump_value,
        avg_inter_jump_interval,
    )


def find_net_jump(signal):
    if not signal:
        return [], {}
    derivative = []
    prev = signal[0]
    for curr in signal[1:]:
        derivative.append(abs(curr - prev))
        prev = curr
    peaks, _ = find_peaks(derivative, threshold=mean(derivative))

    p = {"position": [], "magnitude": []}
    for x in peaks:
        p["position"].append(x)
        p["magnitude"].append(derivative[x])
    return derivative, p


def get_net_stats(val, peaks, derivatives):
    # Calculate average inter jump interval
    intervals = []
    if peaks and peaks["position"]:
        prev = peaks["position"][0]
        for curr in peaks["position"][1:]:
            intervals.append(curr - prev)
            prev = curr
    try:
        sum_load = val[-1] - val[0]
    except IndexError:
        sum_load = 0
    if peaks:
        jump_count = len(peaks["position"])
        if derivatives:
            jump_rate = round(len(peaks["position"]) / len(derivatives), 2)
        else:
            jump_rate = 0
        avg_jump_value = round(mean(peaks["magnitude"]), 2)
        avg_inter_jump_interval = round(mean(intervals), 2)
    else:
        jump_count = 0
        jump_rate = 0
        avg_jump_value = 0
        avg_inter_jump_interval = 0
    return sum_load, jump_count, jump_rate, avg_jump_value, avg_inter_jump_interval


def get_random_users(n=3, stats=True, remove=[]):
    users = []
    while len(users) < n:
        user = random.randint(7, 27)
        if user not in users and user not in remove:
            users.append(user)
    if stats:
        print("Users: {}".format(sorted(users)))
    return users


def get_lda(data, users, comp_num=2, stats=True, score=False):
    data = data[data.user.isin(users)]
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values.ravel()
    z = data.iloc[:, 1].values.ravel()

    lda = LinearDiscriminantAnalysis(n_components=comp_num)
    X_lda = lda.fit_transform(X, y)
    score = round(metrics.calinski_harabasz_score(X_lda, y))
    if stats:
        print(lda.explained_variance_ratio_)
        print(score)

    y = y.reshape(len(y), 1)
    z = z.reshape(len(z), 1)
    df = DataFrame(
        [list(y) + list(z) + list(x) for x, y, z in zip(X_lda, y, z)],
        columns=["user", "try", "component_1", "component_2"],
    )
    if score:
        return df, score
    return df


def get_pca(data, users, comp_num=2, stats=True, score=False):
    data = data[data.user.isin(users)]
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values.ravel()
    z = data.iloc[:, 1].values.ravel()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=comp_num)
    X_pca = pca.fit_transform(X)
    score = round(metrics.calinski_harabasz_score(X_pca, y))
    if stats:
        print(pca.explained_variance_ratio_)
        print(score)

    y = y.reshape(len(y), 1)
    z = z.reshape(len(z), 1)
    df = DataFrame(
        [list(y) + list(z) + list(x) for x, y, z in zip(X_pca, y, z)],
        columns=["user", "try", "component_1", "component_2"],
    )
    if score:
        return df, score
    return df


def calculate_scores(n, datas, epochs=50, verbose=True, norm=False):
    segment_intervals = [10, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    if verbose:
        print("Calculating for {} users.".format(n))
    scores = [[] for _ in range(0, epochs)]
    for i in range(0, epochs):
        users = get_random_users(n=n, stats=False)
        for data in datas:
            _, score = get_lda(data, users, stats=False, score=True)
            scores[i].append(score)
    scores = array(scores).T
    result = []
    max_score = max([mean(x) for x in scores])
    for i in range(0, len(datas)):
        if norm:
            result.append([segment_intervals[i], mean(scores[i]) / max_score, n])
        else:
            result.append([segment_intervals[i], mean(scores[i]), n])
    return DataFrame(result, columns=["time_interval", "score", "users"])


def features_list(index=-1):
    features = [
        "ax_me",
        "ax_sd",
        "ax_mcr",
        "ax_mai",
        "ay_me",
        "ay_sd",
        "ay_mcr",
        "ay_mai",
        "az_me",
        "az_sd",
        "az_mcr",
        "az_mai",
        "a_me",
        "a_sd",
        "a_mcr",
        "a_mai",
        "gx_me",
        "gx_sd",
        "gx_mcr",
        "gy_me",
        "gy_sd",
        "gy_mcr",
        "gz_me",
        "gz_sd",
        "gz_mcr",
        "g_me",
        "g_sd",
        "g_mcr",
        "fa_me",
        "fa_sd",
        "fa_mcr",
        "fb_me",
        "fb_sd",
        "fb_mcr",
        "fc_me",
        "fc_sd",
        "fc_mcr",
        "fd_me",
        "fd_sd",
        "fd_mcr",
        "ca_me",
        "ca_sd",
        "ca_min",
        "ca_max",
        "ca_mcr",
        "cb_me",
        "cb_sd",
        "cb_min",
        "cb_max",
        "cb_mcr",
        "cc_me",
        "cc_sd",
        "cc_min",
        "cc_max",
        "cc_mcr",
        "cd_me",
        "cd_sd",
        "cd_min",
        "cd_max",
        "cd_mcr",
        "c_me",
        "c_sd",
        "c_min",
        "c_max",
        "c_mcr",
        "m_me",
        "m_sd",
        "m_min",
        "m_max",
        "m_jc",
        "m_jr",
        "m_jv",
        "m_iji",
        "ns_me",
        "ns_sd",
        "ns_sum",
        "ns_jc",
        "ns_jr",
        "ns_jv",
        "ns_iji",
        "nr_me",
        "nr_sd",
        "nr_sum",
        "nr_jc",
        "nr_jr",
        "nr_jv",
        "nr_iji",
    ]
    if index == -1:
        return features
    else:
        return features[index]


def show_graphs():
    """
    Not a function that is to be run directly, but an example how to visualize the results.
    This was the best place to put it, to keep it out the way.
    """
    segment_intervals = [10, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    # Parameters
    number_of_users = 7
    segment_interval_length = 60

    # Get n random users and segment interval length
    users = get_random_users(n=number_of_users)
    interval = segment_intervals.index(segment_interval_length)

    datas = []
    data = datas[interval]
    data = data[data.user.isin(users)]
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values.ravel()
    z = data.iloc[:, 1].values.ravel()

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    score = round(metrics.calinski_harabasz_score(X_lda, y))
    y = y.reshape(len(y), 1)
    z = z.reshape(len(z), 1)
    df = DataFrame(
        [list(y) + list(z) + list(x) for x, y, z in zip(X_lda, y, z)],
        columns=["user", "try", "component_1", "component_2"],
    )
    fig = scatter(
        df,
        x="component_1",
        y="component_2",
        color="user",
        color_continuous_scale="Rainbow",
        title="number of users: {}, segment interval: {} seconds, score: {}".format(
            number_of_users, segment_intervals[interval], score
        ),
    )
    fig.show()

    lda = LinearDiscriminantAnalysis(n_components=3)
    X_lda = lda.fit_transform(X, y)
    score = round(metrics.calinski_harabasz_score(X_lda, y))
    y = y.reshape(len(y), 1)
    z = z.reshape(len(z), 1)
    df = DataFrame(
        [list(y) + list(z) + list(x) for x, y, z in zip(X_lda, y, z)],
        columns=["user", "try", "component_1", "component_2", "component_3"],
    )
    fig = scatter_3d(
        df,
        x="component_1",
        y="component_2",
        z="component_3",
        color="user",
        color_continuous_scale="Rainbow",
        title="number of users: {}, segment interval: {} seconds, score: {}".format(
            number_of_users, segment_intervals[interval], score
        ),
    )
    fig.show()


def get_users_data(data):
    """
    Get seance id for each user in a form of a dict.
    """
    users = {}
    for user in list(set(data["user"])):
        x = data[data["user"] == user]
        users.update({user: sorted(list(set(x["seance"])))})
    return users


def get_per_sensor_data(experiment=1, pc_monitor=False):
    segment_intervals = [10, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    sensor_data = []
    com = []
    acc = []
    gyr = []
    frc = []
    cpu = []
    ram = []
    net = []

    for interval in segment_intervals:
        data_name = "jupyter/data/segmented_data_{}_seconds_experiment_{}.csv".format(
            interval, experiment
        )
        data = read_csv(data_name).fillna(0)
        if pc_monitor:
            remain = []
            i = 0
            for x in data[
                [
                    "ca_me",
                    "ca_sd",
                    "ca_min",
                    "ca_max",
                    "ca_mcr",
                    "cb_me",
                    "cb_sd",
                    "cb_min",
                    "cb_max",
                    "cb_mcr",
                    "cc_me",
                    "cc_sd",
                    "cc_min",
                    "cc_max",
                    "cc_mcr",
                    "cd_me",
                    "cd_sd",
                    "cd_min",
                    "cd_max",
                    "cd_mcr",
                    "c_me",
                    "c_sd",
                    "c_min",
                    "c_max",
                    "c_mcr",
                    "m_me",
                    "m_sd",
                    "m_min",
                    "m_max",
                    "m_jc",
                    "m_jr",
                    "m_jv",
                    "m_iji",
                    "ns_me",
                    "ns_sd",
                    "ns_sum",
                    "ns_jc",
                    "ns_jr",
                    "ns_jv",
                    "ns_iji",
                    "nr_me",
                    "nr_sd",
                    "nr_sum",
                    "nr_jc",
                    "nr_jr",
                    "nr_jv",
                    "nr_iji",
                ]
            ].values:
                if mean(x) > 0:
                    remain.append(i)
                i += 1
            data = data.iloc[remain, :]
        label = data.iloc[:, 0:2]
        com.append(data)
        acc.append(label.merge(data.iloc[:, 2:18], left_index=True, right_index=True))
        gyr.append(label.merge(data.iloc[:, 18:30], left_index=True, right_index=True))
        frc.append(label.merge(data.iloc[:, 30:42], left_index=True, right_index=True))
        cpu.append(label.merge(data.iloc[:, 42:67], left_index=True, right_index=True))
        ram.append(label.merge(data.iloc[:, 67:75], left_index=True, right_index=True))
        net.append(label.merge(data.iloc[:, 75:89], left_index=True, right_index=True))
    return {
        "all data": com,
        "accelerometer": acc,
        "gyroscope": gyr,
        "force sensors": frc,
        "cpu": cpu,
        "memory": ram,
        "network": net,
    }


def combine_sensor_data(sensors, names):
    """
    Merge data from multiple sensor.
    """
    return {
        "{} + {}".format(names[0], names[1]): [
            x.merge(y.iloc[:, 2:], left_index=True, right_index=True)
            for x, y in zip(sensors[names[0]], sensors[names[1]])
        ]
    }


def center_graphs():
    from IPython.display import display, HTML
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    init_notebook_mode(connected=True)
    display(
        HTML("""<style>.output_area {display: flex;justify-content: center;}</style>""")
    )


center_graphs()


def process_pir_data(records, start, end, segment_interval, interval=1):
    """
    Classify every time step (determined by interval parameter) into one of categories:
    - moving (sensors, other than pir_01 active)
    - sitting (only pir_01 active)
    - nothing (no sensor activity)
    And calculate some features on top of this classes.
    """
    bins = []
    while start < end:
        bins.append((start, start + timedelta(seconds=interval)))
        start += timedelta(seconds=interval)

    data = []
    active = set()
    activity = 0
    # Iterate through time steps
    for step in bins:
        activity = 0
        # Check for previously active sensors
        if active:
            activity = 2 if max(active) > 1 else 1
        # Iterate through each sensor record in a given time interval
        for record in records.filter(timestamp__range=step).order_by("timestamp"):
            if record.value:
                active.add(int(record.sensor.topic[-1]))
                if int(record.sensor.topic[-1]) > 1:
                    activity = 2
                elif activity < 2:
                    activity = 1
            else:
                try:
                    active.remove(int(record.sensor.topic[-1]))
                except KeyError:
                    # Removing what is already not present
                    pass
        # Add to results
        data.append(activity)
    means = []
    sds = []
    segment_step = int(segment_interval / interval)
    for i in range(0, len(data), segment_step):
        means.append(mean(data[i : i + segment_step]))
        sds.append(std(data[i : i + segment_step]))
    #         # Fill parts where there are zeros between ones - the subject was not moving enough, but still sitting
    #         prev = data[0]
    #         for l in range(1, len(data)):
    #             if prev == 1 and data[i] == 0:
    #                 j = l
    #                 while data[j] == 0:
    #                     j += 1
    #                     if j + 1 >= len(data):
    #                         j -= 1
    #                         break
    #                 if data[j] == 1:
    #                     for k in range(l, j):
    #                         data[k] = 1
    #             prev = data[l]

    #         # Count number of intervals with specific class
    #         class_interval_count = [0, 0, 0]
    #         lengths = []
    #         prev = data[0]
    #         curr_len = 1
    #         for x in data[1:]:
    #             if x != prev:
    #                 class_interval_count[prev] += 1
    #                 if curr_len > 0:
    #                     lengths.append(curr_len)
    #                     curr_len = 1
    #             else:
    #                 curr_len += 1
    #             prev = x
    #         class_interval_count[data[-1]] += 1
    #         lengths.append(curr_len)
    return means, sds


def generate_segmented_data_csv(seconds: int, experiment: int = 1):
    """
    Generate csv file with calculated features, from data subsampled to the given time interval.
    """
    step = timedelta(seconds=seconds)
    seances = Seance.objects.filter(
        experiment__sequence_number=experiment, valid=True
    ).order_by("start")

    print(
        "Generating segmented data csv file with {} seconds intervals for {} seances.".format(
            seconds, seances.count()
        )
    )

    file_name = "segmented_data_{}_seconds_experiment_{}.csv".format(
        seconds, experiment
    )
    with open(file_name, "w") as csv_data:
        csv_data.write(
            "user,seance,ax_me,ax_sd,ax_mcr,ax_mai,ay_me,ay_sd,ay_mcr,ay_mai,az_me,az_sd,az_mcr,az_mai,a_me,"
            + "a_sd,a_mcr,a_mai,gx_me,gx_sd,gx_mcr,gy_me,gy_sd,gy_mcr,gz_me,gz_sd,gz_mcr,g_me,g_sd,g_mcr,fa_me,"
            + "fa_sd,fa_mcr,fb_me,fb_sd,fb_mcr,fc_me,fc_sd,fc_mcr,fd_me,fd_sd,fd_mcr,ca_me,ca_sd,ca_min,ca_max,"
            + "ca_mcr,cb_me,cb_sd,cb_min,cb_max,cb_mcr,cc_me,cc_sd,cc_min,cc_max,cc_mcr,cd_me,cd_sd,cd_min,cd_max,"
            + "cd_mcr,c_me,c_sd,c_min,c_max,c_mcr,m_me,m_sd,m_min,m_max,m_jc,m_jr,m_jv,m_iji,ns_me,ns_sd,ns_sum,"
            + "ns_jc,ns_jr,ns_jv,ns_iji,nr_me,nr_sd,nr_sum,nr_jc,nr_jr,nr_jv,nr_iji,ir_me,ir_sd\n"
        )
        count = 1
        for seance in seances[:]:
            print(
                "-----------------------------------------------------------------------------"
            )
            print("{} of {}".format(count, seances.count()))
            count += 1
            print(seance)
            start = seance.start
            end = seance.end
            data = (
                list(load_data(seance.id, "accelerometer"))
                + list(load_data(seance.id, "gyroscope"))
                + list(load_data(seance.id, "force"))
                + list(load_data(seance.id, "cpu"))
                + [load_data(seance.id, "ram")]
                + list(load_data(seance.id, "net"))
            )
            pir_data = load_data(seance.id, "pir")

            i = 0
            while start < end:
                sub_data = []
                for x in data:
                    try:
                        x[0].sensor
                        sub_data.append(
                            x.filter(timestamp__range=(start, start + step))
                        )
                    except IndexError:
                        sub_data.append([])

                # accelerometer
                ax_val, _, _, ax_me, ax_sd = process_signal(sub_data[0])
                ay_val, _, _, ay_me, ay_sd = process_signal(sub_data[1])
                az_val, _, _, az_me, az_sd = process_signal(sub_data[2])
                a_val, a_me, a_sd = join_accelerometer_signals(ax_val, ay_val, az_val)
                ax_mcr = mean_crossing_rate(ax_val, ax_me)
                ay_mcr = mean_crossing_rate(ay_val, ay_me)
                az_mcr = mean_crossing_rate(az_val, az_me)
                a_mcr = mean_crossing_rate(a_val, a_me)
                ax_mai = mean_acceleration_intensity(ax_val)
                ay_mai = mean_acceleration_intensity(ay_val)
                az_mai = mean_acceleration_intensity(az_val)
                a_mai = mean_acceleration_intensity(a_val)

                # gyroscope
                gx_val, _, _, gx_me, gx_sd = process_signal(sub_data[3])
                gy_val, _, _, gy_me, gy_sd = process_signal(sub_data[4])
                gz_val, _, _, gz_me, gz_sd = process_signal(sub_data[5])
                g_val, g_me, g_sd = join_accelerometer_signals(gx_val, gy_val, gz_val)
                gx_mcr = mean_crossing_rate(gx_val, gx_me)
                gy_mcr = mean_crossing_rate(gy_val, gy_me)
                gz_mcr = mean_crossing_rate(gz_val, gz_me)
                g_mcr = mean_crossing_rate(g_val, g_me)

                # force
                fa_val, _, _, fa_me, fa_sd = process_signal(sub_data[6])
                fb_val, _, _, fb_me, fb_sd = process_signal(sub_data[7])
                fc_val, _, _, fc_me, fc_sd = process_signal(sub_data[8])
                fd_val, _, _, fd_me, fd_sd = process_signal(sub_data[9])
                fa_mcr = mean_crossing_rate(fa_val, fa_me)
                fb_mcr = mean_crossing_rate(fb_val, fb_me)
                fc_mcr = mean_crossing_rate(fc_val, fc_me)
                fd_mcr = mean_crossing_rate(fd_val, fd_me)

                # cpu
                ca_val, _, _, ca_me, ca_sd = process_signal(sub_data[10])
                cb_val, _, _, cb_me, cb_sd = process_signal(sub_data[11])
                cc_val, _, _, cc_me, cc_sd = process_signal(sub_data[12])
                cd_val, _, _, cd_me, cd_sd = process_signal(sub_data[13])
                c_val, c_me, c_sd = join_cpu_signals(ca_val, cb_val, cc_val, cd_val)
                ca_min, ca_max, ca_mcr = get_cpu_stats(ca_val)
                cb_min, cb_max, cb_mcr = get_cpu_stats(cb_val)
                cc_min, cc_max, cc_mcr = get_cpu_stats(cc_val)
                cd_min, cd_max, cd_mcr = get_cpu_stats(cd_val)
                c_min, c_max, c_mcr = get_cpu_stats(c_val)

                # ram
                m_val, _, _, m_me, m_sd = process_signal(sub_data[14])
                derivatives, peaks = find_ram_jump(m_val)
                m_me, m_min, m_max, m_jc, m_jr, m_jv, m_iji = get_mem_stats(
                    m_val, peaks, derivatives
                )

                # net
                ns_val, _, _, ns_me, ns_sd = process_signal(sub_data[15])
                nr_val, _, _, nr_me, nr_sd = process_signal(sub_data[16])
                ns_der, ns_pe = find_net_jump(ns_val)
                nr_der, nr_pe = find_net_jump(nr_val)
                ns_sum, ns_jc, ns_jr, ns_jv, ns_iji = get_net_stats(
                    ns_val, ns_pe, ns_der
                )
                nr_sum, nr_jc, nr_jr, nr_jv, nr_iji = get_net_stats(
                    nr_val, nr_pe, nr_der
                )

                # ir sensors
                pir_datas = process_pir_data(
                    pir_data, seance.start, seance.end, seconds, interval=5
                )
                ir_me = pir_datas[0][i]
                ir_sd = pir_datas[1][i]

                write_row = ",".join(
                    [
                        str(x)
                        for x in [
                            seance.user.id,
                            seance.id,
                            ax_me,
                            ax_sd,
                            ax_mcr,
                            ax_mai,
                            ay_me,
                            ay_sd,
                            ay_mcr,
                            ay_mai,
                            az_me,
                            az_sd,
                            az_mcr,
                            az_mai,
                            a_me,
                            a_sd,
                            a_mcr,
                            a_mai,
                            gx_me,
                            gx_sd,
                            gx_mcr,
                            gy_me,
                            gy_sd,
                            gy_mcr,
                            gz_me,
                            gz_sd,
                            gz_mcr,
                            g_me,
                            g_sd,
                            g_mcr,
                            fa_me,
                            fa_sd,
                            fa_mcr,
                            fb_me,
                            fb_sd,
                            fb_mcr,
                            fc_me,
                            fc_sd,
                            fc_mcr,
                            fd_me,
                            fd_sd,
                            fd_mcr,
                            ca_me,
                            ca_sd,
                            ca_min,
                            ca_max,
                            ca_mcr,
                            cb_me,
                            cb_sd,
                            cb_min,
                            cb_max,
                            cb_mcr,
                            cc_me,
                            cc_sd,
                            cc_min,
                            cc_max,
                            cc_mcr,
                            cd_me,
                            cd_sd,
                            cd_min,
                            cd_max,
                            cd_mcr,
                            c_me,
                            c_sd,
                            c_min,
                            c_max,
                            c_mcr,
                            m_me,
                            m_sd,
                            m_min,
                            m_max,
                            m_jc,
                            m_jr,
                            m_jv,
                            m_iji,
                            ns_me,
                            ns_sd,
                            ns_sum,
                            ns_jc,
                            ns_jr,
                            ns_jv,
                            ns_iji,
                            nr_me,
                            nr_sd,
                            nr_sum,
                            nr_jc,
                            nr_jr,
                            nr_jv,
                            nr_iji,
                            ir_me,
                            ir_sd,
                        ]
                    ]
                )
                csv_data.write(write_row + "\n")
                start += step
                i += 1


def generate_segmented_data_pc_monitor(seconds, experiment):
    step = timedelta(seconds=seconds)
    seances = Seance.objects.filter(
        experiment__sequence_number=experiment, valid=True
    ).order_by("start")

    df = {
        "user": [],
        "seance": [],
        "ax_me": [],
        "ax_sd": [],
        "ax_mcr": [],
        "ax_mai": [],
        "ay_me": [],
        "ay_sd": [],
        "ay_mcr": [],
        "ay_mai": [],
        "az_me": [],
        "az_sd": [],
        "az_mcr": [],
        "az_mai": [],
        "a_me": [],
        "a_sd": [],
        "a_mcr": [],
        "a_mai": [],
        "gx_me": [],
        "gx_sd": [],
        "gx_mcr": [],
        "gy_me": [],
        "gy_sd": [],
        "gy_mcr": [],
        "gz_me": [],
        "gz_sd": [],
        "gz_mcr": [],
        "g_me": [],
        "g_sd": [],
        "g_mcr": [],
        "fa_me": [],
        "fa_sd": [],
        "fa_mcr": [],
        "fb_me": [],
        "fb_sd": [],
        "fb_mcr": [],
        "fc_me": [],
        "fc_sd": [],
        "fc_mcr": [],
        "fd_me": [],
        "fd_sd": [],
        "fd_mcr": [],
        "ca_me": [],
        "ca_sd": [],
        "ca_min": [],
        "ca_max": [],
        "ca_mcr": [],
        "cb_me": [],
        "cb_sd": [],
        "cb_min": [],
        "cb_max": [],
        "cb_mcr": [],
        "cc_me": [],
        "cc_sd": [],
        "cc_min": [],
        "cc_max": [],
        "cc_mcr": [],
        "cd_me": [],
        "cd_sd": [],
        "cd_min": [],
        "cd_max": [],
        "cd_mcr": [],
        "c_me": [],
        "c_sd": [],
        "c_min": [],
        "c_max": [],
        "c_mcr": [],
        "m_me": [],
        "m_sd": [],
        "m_min": [],
        "m_max": [],
        "m_jc": [],
        "m_jr": [],
        "m_jv": [],
        "m_iji": [],
        "ns_me": [],
        "ns_sd": [],
        "ns_sum": [],
        "ns_jc": [],
        "ns_jr": [],
        "ns_jv": [],
        "ns_iji": [],
        "nr_me": [],
        "nr_sd": [],
        "nr_sum": [],
        "nr_jc": [],
        "nr_jr": [],
        "nr_jv": [],
        "nr_iji": [],
        "ir_me": [],
    }
    print(
        "Generating segmented data csv file with {} seconds intervals for {} seances.".format(
            seconds, seances.count()
        )
    )
    count = 1
    bad_users = []
    for seance in seances:
        print(
            "-----------------------------------------------------------------------------"
        )
        print("{} of {}".format(count, seances.count()))
        count += 1
        print(seance)
        if seance.user.id in bad_users:
            print("Bad user, skipping...")
            continue
        data = (
            list(load_data(seance.id, "accelerometer"))
            + list(load_data(seance.id, "gyroscope"))
            + list(load_data(seance.id, "force"))
            + list(load_data(seance.id, "cpu"))
            + [load_data(seance.id, "ram")]
            + list(load_data(seance.id, "net"))
        )
        pir_data = load_data(seance.id, "pir")
        pir_datas = process_pir_data(
            pir_data, seance.start, seance.end, seconds, interval=10
        )
        try:
            start = data[14].order_by("timestamp")[0].timestamp
            end = data[14].order_by("-timestamp")[0].timestamp
        except IndexError:
            print("BAD USER... REMOVING FROM RESULTS")
            bad_users.append(seance.user.id)
            continue
        i = 0
        while start < end:
            sub_data = []
            for x in data:
                try:
                    x[0].sensor
                    sub_data.append(x.filter(timestamp__range=(start, start + step)))
                except IndexError:
                    sub_data.append([])

            # accelerometer
            ax_val, _, _, ax_me, ax_sd = process_signal(sub_data[0])
            ay_val, _, _, ay_me, ay_sd = process_signal(sub_data[1])
            az_val, _, _, az_me, az_sd = process_signal(sub_data[2])
            a_val, a_me, a_sd = join_accelerometer_signals(ax_val, ay_val, az_val)
            ax_mcr = mean_crossing_rate(ax_val, ax_me)
            ay_mcr = mean_crossing_rate(ay_val, ay_me)
            az_mcr = mean_crossing_rate(az_val, az_me)
            a_mcr = mean_crossing_rate(a_val, a_me)
            ax_mai = mean_acceleration_intensity(ax_val)
            ay_mai = mean_acceleration_intensity(ay_val)
            az_mai = mean_acceleration_intensity(az_val)
            a_mai = mean_acceleration_intensity(a_val)

            # gyroscope
            gx_val, _, _, gx_me, gx_sd = process_signal(sub_data[3])
            gy_val, _, _, gy_me, gy_sd = process_signal(sub_data[4])
            gz_val, _, _, gz_me, gz_sd = process_signal(sub_data[5])
            g_val, g_me, g_sd = join_accelerometer_signals(gx_val, gy_val, gz_val)
            gx_mcr = mean_crossing_rate(gx_val, gx_me)
            gy_mcr = mean_crossing_rate(gy_val, gy_me)
            gz_mcr = mean_crossing_rate(gz_val, gz_me)
            g_mcr = mean_crossing_rate(g_val, g_me)

            # force
            fa_val, _, _, fa_me, fa_sd = process_signal(sub_data[6])
            fb_val, _, _, fb_me, fb_sd = process_signal(sub_data[7])
            fc_val, _, _, fc_me, fc_sd = process_signal(sub_data[8])
            fd_val, _, _, fd_me, fd_sd = process_signal(sub_data[9])
            fa_mcr = mean_crossing_rate(fa_val, fa_me)
            fb_mcr = mean_crossing_rate(fb_val, fb_me)
            fc_mcr = mean_crossing_rate(fc_val, fc_me)
            fd_mcr = mean_crossing_rate(fd_val, fd_me)

            # cpu
            ca_val, _, _, ca_me, ca_sd = process_signal(sub_data[10])
            cb_val, _, _, cb_me, cb_sd = process_signal(sub_data[11])
            cc_val, _, _, cc_me, cc_sd = process_signal(sub_data[12])
            cd_val, _, _, cd_me, cd_sd = process_signal(sub_data[13])
            c_val, c_me, c_sd = join_cpu_signals(ca_val, cb_val, cc_val, cd_val)
            ca_min, ca_max, ca_mcr = get_cpu_stats(ca_val)
            cb_min, cb_max, cb_mcr = get_cpu_stats(cb_val)
            cc_min, cc_max, cc_mcr = get_cpu_stats(cc_val)
            cd_min, cd_max, cd_mcr = get_cpu_stats(cd_val)
            c_min, c_max, c_mcr = get_cpu_stats(c_val)

            # ram
            m_val, _, _, m_me, m_sd = process_signal(sub_data[14])
            derivatives, peaks = find_ram_jump(m_val)
            m_me, m_min, m_max, m_jc, m_jr, m_jv, m_iji = get_mem_stats(
                m_val, peaks, derivatives
            )

            # net
            ns_val, _, _, ns_me, ns_sd = process_signal(sub_data[15])
            nr_val, _, _, nr_me, nr_sd = process_signal(sub_data[16])
            ns_der, ns_pe = find_net_jump(ns_val)
            nr_der, nr_pe = find_net_jump(nr_val)
            ns_sum, ns_jc, ns_jr, ns_jv, ns_iji = get_net_stats(ns_val, ns_pe, ns_der)
            nr_sum, nr_jc, nr_jr, nr_jv, nr_iji = get_net_stats(nr_val, nr_pe, nr_der)

            # ir sensors
            ir_me = pir_datas[i]

            df["user"].append(seance.user.id)
            df["seance"].append(seance.id)
            df["ax_me"].append(ax_me)
            df["ax_sd"].append(ax_sd)
            df["ax_mcr"].append(ax_mcr)
            df["ax_mai"].append(ax_mai)
            df["ay_me"].append(ay_me)
            df["ay_sd"].append(ay_sd)
            df["ay_mcr"].append(ay_mcr)
            df["ay_mai"].append(ay_mai)
            df["az_me"].append(az_me)
            df["az_sd"].append(az_sd)
            df["az_mcr"].append(az_mcr)
            df["az_mai"].append(az_mai)
            df["a_me"].append(a_me)
            df["a_sd"].append(a_sd)
            df["a_mcr"].append(a_mcr)
            df["a_mai"].append(a_mai)
            df["gx_me"].append(gx_me)
            df["gx_sd"].append(gx_sd)
            df["gx_mcr"].append(gx_mcr)
            df["gy_me"].append(gy_me)
            df["gy_sd"].append(gy_sd)
            df["gy_mcr"].append(gy_mcr)
            df["gz_me"].append(gz_me)
            df["gz_sd"].append(gz_sd)
            df["gz_mcr"].append(gz_mcr)
            df["g_me"].append(g_me)
            df["g_sd"].append(g_sd)
            df["g_mcr"].append(g_mcr)
            df["fa_me"].append(fa_me)
            df["fa_sd"].append(fa_sd)
            df["fa_mcr"].append(fa_mcr)
            df["fb_me"].append(fb_me)
            df["fb_sd"].append(fb_sd)
            df["fb_mcr"].append(fb_mcr)
            df["fc_me"].append(fc_me)
            df["fc_sd"].append(fc_sd)
            df["fc_mcr"].append(fc_mcr)
            df["fd_me"].append(fd_me)
            df["fd_sd"].append(fd_sd)
            df["fd_mcr"].append(fd_mcr)
            df["ca_me"].append(ca_me)
            df["ca_sd"].append(ca_sd)
            df["ca_min"].append(ca_min)
            df["ca_max"].append(ca_max)
            df["ca_mcr"].append(ca_mcr)
            df["cb_me"].append(cb_me)
            df["cb_sd"].append(cb_sd)
            df["cb_min"].append(cb_min)
            df["cb_max"].append(cb_max)
            df["cb_mcr"].append(cb_mcr)
            df["cc_me"].append(cc_me)
            df["cc_sd"].append(cc_sd)
            df["cc_min"].append(cc_min)
            df["cc_max"].append(cc_max)
            df["cc_mcr"].append(cc_mcr)
            df["cd_me"].append(cd_me)
            df["cd_sd"].append(cd_sd)
            df["cd_min"].append(cd_min)
            df["cd_max"].append(cd_max)
            df["cd_mcr"].append(cd_mcr)
            df["c_me"].append(c_me)
            df["c_sd"].append(c_sd)
            df["c_min"].append(c_min)
            df["c_max"].append(c_max)
            df["c_mcr"].append(c_mcr)
            df["m_me"].append(m_me)
            df["m_sd"].append(m_sd)
            df["m_min"].append(m_min)
            df["m_max"].append(m_max)
            df["m_jc"].append(m_jc)
            df["m_jr"].append(m_jr)
            df["m_jv"].append(m_jv)
            df["m_iji"].append(m_iji)
            df["ns_me"].append(ns_me)
            df["ns_sd"].append(ns_sd)
            df["ns_sum"].append(ns_sum)
            df["ns_jc"].append(ns_jc)
            df["ns_jr"].append(ns_jr)
            df["ns_jv"].append(ns_jv)
            df["ns_iji"].append(ns_iji)
            df["nr_me"].append(nr_me)
            df["nr_sd"].append(nr_sd)
            df["nr_sum"].append(nr_sum)
            df["nr_jc"].append(nr_jc)
            df["nr_jr"].append(nr_jr)
            df["nr_jv"].append(nr_jv)
            df["nr_iji"].append(nr_iji)
            df["ir_me"].append(ir_me)

            start += step
            i += 1

    # Write the dataframe to a csv file
    file_name = "segmented_data_{}_seconds_experiment_{}_pc_monitor.csv".format(
        seconds, experiment
    )
    df = DataFrame(df)
    for x in bad_users:
        df = df[df["user"] != x]
    df.to_csv(file_name, index=False)


def generate_segmented_frequency_data_csv(seconds: int, experiment: int = 1):
    """
    Generate csv file with calculated features, from data subsampled to the given time interval.
    """

    def get_10_mag_bins(topic_records):
        """
        Calculate 10 magnitude bins, based on frequencies
        """
        if not len(topic_records):
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        values = [x.value for x in topic_records]
        tim = [x.timestamp for x in topic_records]
        step = 5
        values_inter, tim_inter = interpolate(topic_records, step)
        if not values_inter:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Frequency domain analysis
        n = len(values_inter)
        y = fft(values_inter)
        f = fftfreq(len(y))
        # Multiply frequency by sampling rate, to obtain Hertz
        f = f * (1000 / step)

        # Take only first half of the fft
        y = y[1 : len(y) // 2]
        f = f[1 : len(f) // 2]
        mag = [sqrt(r ** 2 + i ** 2) for r, i in zip(y.real, y.imag)]

        prev = 0
        step = 10
        f = list(f)
        squashed = {"freq": [], "value": []}
        for freq_step in range(step, 101, step):
            freq = [x for x in f if prev < x < freq_step]
            indices = [f.index(x) for x in freq]
            prev = freq_step
            squashed["freq"].append((prev + freq_step) / 2)
            squashed["value"].append(mean([mag[x] for x in indices]))
        return squashed["value"]

    step = timedelta(seconds=seconds)
    seances = Seance.objects.filter(
        experiment__sequence_number=experiment, valid=True
    ).order_by("start")

    print(
        "Generating segmented data csv file with {} seconds intervals for {} seances.".format(
            seconds, seances.count()
        )
    )

    file_name = "segmented_data_{}_seconds_experiment_{}_frequency.csv".format(
        seconds, experiment
    )
    resulting_data = {
        "user": [],
        "seance": [],
        "ax_me": [],
        "ax_sd": [],
        "ax_mcr": [],
        "ax_mai": [],
        "ay_me": [],
        "ay_sd": [],
        "ay_mcr": [],
        "ay_mai": [],
        "az_me": [],
        "az_sd": [],
        "az_mcr": [],
        "az_mai": [],
        "a_me": [],
        "a_sd": [],
        "a_mcr": [],
        "a_mai": [],
        "gx_me": [],
        "gx_sd": [],
        "gx_mcr": [],
        "gy_me": [],
        "gy_sd": [],
        "gy_mcr": [],
        "gz_me": [],
        "gz_sd": [],
        "gz_mcr": [],
        "g_me": [],
        "g_sd": [],
        "g_mcr": [],
        "fa_me": [],
        "fa_sd": [],
        "fa_mcr": [],
        "fb_me": [],
        "fb_sd": [],
        "fb_mcr": [],
        "fc_me": [],
        "fc_sd": [],
        "fc_mcr": [],
        "fd_me": [],
        "fd_sd": [],
        "fd_mcr": [],
        "ca_me": [],
        "ca_sd": [],
        "ca_min": [],
        "ca_max": [],
        "ca_mcr": [],
        "cb_me": [],
        "cb_sd": [],
        "cb_min": [],
        "cb_max": [],
        "cb_mcr": [],
        "cc_me": [],
        "cc_sd": [],
        "cc_min": [],
        "cc_max": [],
        "cc_mcr": [],
        "cd_me": [],
        "cd_sd": [],
        "cd_min": [],
        "cd_max": [],
        "cd_mcr": [],
        "c_me": [],
        "c_sd": [],
        "c_min": [],
        "c_max": [],
        "c_mcr": [],
        "m_me": [],
        "m_sd": [],
        "m_min": [],
        "m_max": [],
        "m_jc": [],
        "m_jr": [],
        "m_jv": [],
        "m_iji": [],
        "ns_me": [],
        "ns_sd": [],
        "ns_sum": [],
        "ns_jc": [],
        "ns_jr": [],
        "ns_jv": [],
        "ns_iji": [],
        "nr_me": [],
        "nr_sd": [],
        "nr_sum": [],
        "nr_jc": [],
        "nr_jr": [],
        "nr_jv": [],
        "nr_iji": [],
        "ir_me": [],
        "ax_bin": [],
        "ay_bin": [],
        "az_bin": [],
        "gx_bin": [],
        "gy_bin": [],
        "gz_bin": [],
        "fa_bin": [],
        "fb_bin": [],
        "fc_bin": [],
        "fd_bin": [],
    }

    count = 1
    for seance in seances[:]:
        print(
            "-----------------------------------------------------------------------------"
        )
        print("{} of {}".format(count, seances.count()))
        count += 1
        print(seance)
        start = seance.start
        end = seance.end
        data = (
            list(load_data(seance.id, "accelerometer"))
            + list(load_data(seance.id, "gyroscope"))
            + list(load_data(seance.id, "force"))
            + list(load_data(seance.id, "cpu"))
            + [load_data(seance.id, "ram")]
            + list(load_data(seance.id, "net"))
        )
        pir_data = load_data(seance.id, "pir")
        pir_datas = process_pir_data(
            pir_data, seance.start, seance.end, seconds, interval=10
        )

        i = 0
        while start < end:
            sub_data = []
            for x in data:
                try:
                    x[0].sensor
                    sub_data.append(x.filter(timestamp__range=(start, start + step)))
                except IndexError:
                    sub_data.append([])

            # accelerometer
            ax_val, _, _, ax_me, ax_sd = process_signal(sub_data[0])
            ay_val, _, _, ay_me, ay_sd = process_signal(sub_data[1])
            az_val, _, _, az_me, az_sd = process_signal(sub_data[2])
            a_val, a_me, a_sd = join_accelerometer_signals(ax_val, ay_val, az_val)
            ax_mcr = mean_crossing_rate(ax_val, ax_me)
            ay_mcr = mean_crossing_rate(ay_val, ay_me)
            az_mcr = mean_crossing_rate(az_val, az_me)
            a_mcr = mean_crossing_rate(a_val, a_me)
            ax_mai = mean_acceleration_intensity(ax_val)
            ay_mai = mean_acceleration_intensity(ay_val)
            az_mai = mean_acceleration_intensity(az_val)
            a_mai = mean_acceleration_intensity(a_val)
            ax_bin = get_10_mag_bins(sub_data[0])
            ay_bin = get_10_mag_bins(sub_data[1])
            az_bin = get_10_mag_bins(sub_data[2])

            # gyroscope
            gx_val, _, _, gx_me, gx_sd = process_signal(sub_data[3])
            gy_val, _, _, gy_me, gy_sd = process_signal(sub_data[4])
            gz_val, _, _, gz_me, gz_sd = process_signal(sub_data[5])
            g_val, g_me, g_sd = join_accelerometer_signals(gx_val, gy_val, gz_val)
            gx_mcr = mean_crossing_rate(gx_val, gx_me)
            gy_mcr = mean_crossing_rate(gy_val, gy_me)
            gz_mcr = mean_crossing_rate(gz_val, gz_me)
            g_mcr = mean_crossing_rate(g_val, g_me)
            gx_bin = get_10_mag_bins(sub_data[3])
            gy_bin = get_10_mag_bins(sub_data[4])
            gz_bin = get_10_mag_bins(sub_data[5])

            # force
            fa_val, _, _, fa_me, fa_sd = process_signal(sub_data[6])
            fb_val, _, _, fb_me, fb_sd = process_signal(sub_data[7])
            fc_val, _, _, fc_me, fc_sd = process_signal(sub_data[8])
            fd_val, _, _, fd_me, fd_sd = process_signal(sub_data[9])
            fa_mcr = mean_crossing_rate(fa_val, fa_me)
            fb_mcr = mean_crossing_rate(fb_val, fb_me)
            fc_mcr = mean_crossing_rate(fc_val, fc_me)
            fd_mcr = mean_crossing_rate(fd_val, fd_me)
            fa_bin = get_10_mag_bins(sub_data[6])
            fb_bin = get_10_mag_bins(sub_data[7])
            fc_bin = get_10_mag_bins(sub_data[8])
            fd_bin = get_10_mag_bins(sub_data[9])

            # cpu
            ca_val, _, _, ca_me, ca_sd = process_signal(sub_data[10])
            cb_val, _, _, cb_me, cb_sd = process_signal(sub_data[11])
            cc_val, _, _, cc_me, cc_sd = process_signal(sub_data[12])
            cd_val, _, _, cd_me, cd_sd = process_signal(sub_data[13])
            c_val, c_me, c_sd = join_cpu_signals(ca_val, cb_val, cc_val, cd_val)
            ca_min, ca_max, ca_mcr = get_cpu_stats(ca_val)
            cb_min, cb_max, cb_mcr = get_cpu_stats(cb_val)
            cc_min, cc_max, cc_mcr = get_cpu_stats(cc_val)
            cd_min, cd_max, cd_mcr = get_cpu_stats(cd_val)
            c_min, c_max, c_mcr = get_cpu_stats(c_val)

            # ram
            m_val, _, _, m_me, m_sd = process_signal(sub_data[14])
            derivatives, peaks = find_ram_jump(m_val)
            m_me, m_min, m_max, m_jc, m_jr, m_jv, m_iji = get_mem_stats(
                m_val, peaks, derivatives
            )

            # net
            ns_val, _, _, ns_me, ns_sd = process_signal(sub_data[15])
            nr_val, _, _, nr_me, nr_sd = process_signal(sub_data[16])
            ns_der, ns_pe = find_net_jump(ns_val)
            nr_der, nr_pe = find_net_jump(nr_val)
            ns_sum, ns_jc, ns_jr, ns_jv, ns_iji = get_net_stats(ns_val, ns_pe, ns_der)
            nr_sum, nr_jc, nr_jr, nr_jv, nr_iji = get_net_stats(nr_val, nr_pe, nr_der)

            # ir sensors
            ir_me = pir_datas[i]

            resulting_data["user"].append(seance.user.id)
            resulting_data["seance"].append(seance.id)
            resulting_data["ax_me"].append(ax_me)
            resulting_data["ax_sd"].append(ax_sd)
            resulting_data["ax_mcr"].append(ax_mcr)
            resulting_data["ax_mai"].append(ax_mai)
            resulting_data["ay_me"].append(ay_me)
            resulting_data["ay_sd"].append(ay_sd)
            resulting_data["ay_mcr"].append(ay_mcr)
            resulting_data["ay_mai"].append(ay_mai)
            resulting_data["az_me"].append(az_me)
            resulting_data["az_sd"].append(az_sd)
            resulting_data["az_mcr"].append(az_mcr)
            resulting_data["az_mai"].append(az_mai)
            resulting_data["a_me"].append(a_me)
            resulting_data["a_sd"].append(a_sd)
            resulting_data["a_mcr"].append(a_mcr)
            resulting_data["a_mai"].append(a_mai)
            resulting_data["gx_me"].append(gx_me)
            resulting_data["gx_sd"].append(gx_sd)
            resulting_data["gx_mcr"].append(gx_mcr)
            resulting_data["gy_me"].append(gy_me)
            resulting_data["gy_sd"].append(gy_sd)
            resulting_data["gy_mcr"].append(gy_mcr)
            resulting_data["gz_me"].append(gz_me)
            resulting_data["gz_sd"].append(gz_sd)
            resulting_data["gz_mcr"].append(gz_mcr)
            resulting_data["g_me"].append(g_me)
            resulting_data["g_sd"].append(g_sd)
            resulting_data["g_mcr"].append(g_mcr)
            resulting_data["fa_me"].append(fa_me)
            resulting_data["fa_sd"].append(fa_sd)
            resulting_data["fa_mcr"].append(fa_mcr)
            resulting_data["fb_me"].append(fb_me)
            resulting_data["fb_sd"].append(fb_sd)
            resulting_data["fb_mcr"].append(fb_mcr)
            resulting_data["fc_me"].append(fc_me)
            resulting_data["fc_sd"].append(fc_sd)
            resulting_data["fc_mcr"].append(fc_mcr)
            resulting_data["fd_me"].append(fd_me)
            resulting_data["fd_sd"].append(fd_sd)
            resulting_data["fd_mcr"].append(fd_mcr)
            resulting_data["ca_me"].append(ca_me)
            resulting_data["ca_sd"].append(ca_sd)
            resulting_data["ca_min"].append(ca_min)
            resulting_data["ca_max"].append(ca_max)
            resulting_data["ca_mcr"].append(ca_mcr)
            resulting_data["cb_me"].append(cb_me)
            resulting_data["cb_sd"].append(cb_sd)
            resulting_data["cb_min"].append(cb_min)
            resulting_data["cb_max"].append(cb_max)
            resulting_data["cb_mcr"].append(cb_mcr)
            resulting_data["cc_me"].append(cc_me)
            resulting_data["cc_sd"].append(cc_sd)
            resulting_data["cc_min"].append(cc_min)
            resulting_data["cc_max"].append(cc_max)
            resulting_data["cc_mcr"].append(cc_mcr)
            resulting_data["cd_me"].append(cd_me)
            resulting_data["cd_sd"].append(cd_sd)
            resulting_data["cd_min"].append(cd_min)
            resulting_data["cd_max"].append(cd_max)
            resulting_data["cd_mcr"].append(cd_mcr)
            resulting_data["c_me"].append(c_me)
            resulting_data["c_sd"].append(c_sd)
            resulting_data["c_min"].append(c_min)
            resulting_data["c_max"].append(c_max)
            resulting_data["c_mcr"].append(c_mcr)
            resulting_data["m_me"].append(m_me)
            resulting_data["m_sd"].append(m_sd)
            resulting_data["m_min"].append(m_min)
            resulting_data["m_max"].append(m_max)
            resulting_data["m_jc"].append(m_jc)
            resulting_data["m_jr"].append(m_jr)
            resulting_data["m_jv"].append(m_jv)
            resulting_data["m_iji"].append(m_iji)
            resulting_data["ns_me"].append(ns_me)
            resulting_data["ns_sd"].append(ns_sd)
            resulting_data["ns_sum"].append(ns_sum)
            resulting_data["ns_jc"].append(ns_jc)
            resulting_data["ns_jr"].append(ns_jr)
            resulting_data["ns_jv"].append(ns_jv)
            resulting_data["ns_iji"].append(ns_iji)
            resulting_data["nr_me"].append(nr_me)
            resulting_data["nr_sd"].append(nr_sd)
            resulting_data["nr_sum"].append(nr_sum)
            resulting_data["nr_jc"].append(nr_jc)
            resulting_data["nr_jr"].append(nr_jr)
            resulting_data["nr_jv"].append(nr_jv)
            resulting_data["nr_iji"].append(nr_iji)
            resulting_data["ir_me"].append(ir_me)
            resulting_data["ax_bin"].append(ax_bin)
            resulting_data["ay_bin"].append(ay_bin)
            resulting_data["az_bin"].append(az_bin)
            resulting_data["gx_bin"].append(gx_bin)
            resulting_data["gy_bin"].append(gy_bin)
            resulting_data["gz_bin"].append(gz_bin)
            resulting_data["fa_bin"].append(fa_bin)
            resulting_data["fb_bin"].append(fb_bin)
            resulting_data["fc_bin"].append(fc_bin)
            resulting_data["fd_bin"].append(fd_bin)

            start += step
            i += 1
    df = DataFrame(resulting_data)
    df.to_csv(file_name, index=False)


def interpolate(data, step: int):
    """
    Interpolate list of sensor records, to the given milisecond interval.
    Return interpolated values and associated times.
    """
    data = sorted(data, key=lambda x: x.timestamp)
    current_step = data[0].timestamp + timedelta(microseconds=step * 1000)
    values = [x.value for x in data]
    tim = [datetime.timestamp(x.timestamp) for x in data]
    try:
        intepolator = interp1d(tim, values)
    except ValueError:
        return None, None
    values_inter = []
    tim_inter = []
    while current_step <= data[-1].timestamp:
        values_inter.append(intepolator(datetime.timestamp(current_step)))
        tim_inter.append(current_step)
        current_step += timedelta(microseconds=step * 1000)
    return values_inter, tim_inter


def calculate_f1_on_preds(userss, all_predictions, y_tests, verbose=False):
    # Calculate f1 score per classifiers on raw model predictions
    f1_scores = []
    for j in range(len(all_predictions)):
        users = userss[j]
        predictions = all_predictions[j]
        y_test = y_tests[j]
        row = []
        for i in range(len(users)):
            user = users[i]
            y_mod = [1 if x == user else 0 for x in y_test]
            row.append(metrics.f1_score(y_mod, predictions[i]))
        f1_scores.append(row)
    if verbose:
        print("userss = {}".format(userss))
        print("f1_scores = {}".format(f1_scores))
    fig = go.Figure(
        data=[
            go.Bar(x=userss[i], y=f1_scores[i], name="Task {}".format(i + 1))
            for i in range(len(userss))
        ]
    )
    fig.update_layout(
        title="Per user f1 scores",
        xaxis_title="user",
        yaxis_title="f1 score",
        xaxis_type="category",
        barmode="group",
    )
    fig.show()


def get_probs_per_test_case(userss, all_probabilities, y_tests, verbose=False):
    """
    Predict every test sample into the class with highest probability
    if there is more than one class with the highest probability,
    mark it as none of the users.
    """
    all_preds = []
    k = 0
    for probabilities in all_probabilities:
        users = userss[k]
        y_test = y_tests[k]
        k += 1
        case_probs = []
        for i in range(len(probabilities[0])):
            case = {}
            for j in range(len(probabilities)):
                case.update({str(users[j]): probabilities[j][i]})
            case_probs.append(case)

        probs = []
        y_pred = []
        for i in range(len(case_probs)):
            max_val = 0
            case = case_probs[i]
            for key in case:
                if case[key][1] > max_val:
                    max_val = case[key][1]
            preds = [x for x in case if case[x][1] == max_val]
            if len(preds) > 1:
                preds = -1
            else:
                preds = preds[0]
            probs.append([int(y_test[i]), int(preds), max_val])
            y_pred.append(int(preds))
        preds = []
        for user in users:
            y_tmp_test = []
            y_tmp_pred = []
            for i in range(len(y_test)):
                if y_test[i] == user:
                    y_tmp_test.append(y_test[i])
                    y_tmp_pred.append(y_pred[i])
            preds.append(metrics.accuracy_score(y_tmp_test, y_tmp_pred))
        all_preds.append(preds)
    if verbose:
        print("userss = {}".format(userss))
        print("all_preds = {}".format(all_preds))
    fig = go.Figure(
        data=[
            go.Bar(x=userss[i], y=all_preds[i], name="Task {}".format(i + 1))
            for i in range(len(all_preds))
        ]
    )
    fig.update_layout(
        title="Per user classification accuracy",
        xaxis_title="user",
        yaxis_title="classification accuracy",
        xaxis_type="category",
    )
    fig.show()


def get_probs_per_test_case_single_task(users, probabilities, y_test):
    """
    Predict every test sample into the class with highest probability
    if there is more than one class with the highest probability,
    mark it as none of the users.
    """
    case_probs = []
    for i in range(len(probabilities[0])):
        case = {}
        for j in range(len(probabilities)):
            case.update({str(users[j]): probabilities[j][i]})
        case_probs.append(case)
    probs = []
    y_pred = []
    for i in range(len(case_probs)):
        max_val = 0
        case = case_probs[i]
        for key in case:
            if case[key][1] > max_val:
                max_val = case[key][1]
        preds = [x for x in case if case[x][1] == max_val]
        if len(preds) > 1:
            preds = -1
        else:
            preds = preds[0]
        probs.append([int(y_test[i]), int(preds), max_val])
        y_pred.append(int(preds))

    preds = []
    for user in users:
        y_tmp_test = []
        y_tmp_pred = []
        for i in range(len(y_test)):
            if y_test[i] == user:
                y_tmp_test.append(y_test[i])
                y_tmp_pred.append(y_pred[i])
        preds.append(metrics.accuracy_score(y_tmp_test, y_tmp_pred))
    return preds
