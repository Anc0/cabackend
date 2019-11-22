from datetime import datetime, timedelta

import pytz
from scipy.interpolate import interp1d


class LinearInterpolation:
    def __init__(self, data):
        """
        :param data: list of dicts wth timestamp and value attributes
        """
        self.data = sorted(data, key=lambda x: x.timestamp)

        self.interpolated_data = []

    def interpolate(self, step):
        """
        Interpolate the data from the first data timestamp, to the last possible step, smaller than the last timestamp.
        :param step: interpolation step, set in miliseconds.
        """
        # Firstly reset the interpolated values.
        self.interpolated_data = []
        current_step = self.data[0]["timestamp"] + timedelta(microseconds=step * 1000)

        timestamps = [datetime.timestamp(x["timestamp"]) for x in self.data]
        values = [x["value"] for x in self.data]
        intepolator = interp1d(timestamps, values)
        while current_step <= self.data[-1]["timestamp"]:
            print(intepolator(datetime.timestamp(current_step)))

            current_step += timedelta(microseconds=step * 1000)
