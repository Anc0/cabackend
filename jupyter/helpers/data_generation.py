from datetime import timedelta, datetime
from math import sqrt

from django.db.models import Q
from numpy import mean, std, array
from numpy.fft import fft, fftfreq
from pandas import DataFrame, read_csv
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from seances.models import Seance
from sensors.models import Sensor, SensorRecord


class SeanceHandler:
    def __init__(self, task_number: int):
        """
        Class that provides function that are (per task) seance related.
        """
        self.task_number = task_number
        self.seance_ids = self._get_seance_ids()

    def get_valid_seances(self):
        """
        Return seance ids, based on the seance validity check.
        """
        seances = Seance.objects.filter(id__in=self.seance_ids).order_by("created")
        invalid_users = []

        for seance in seances:
            if not self.is_valid_seance(seance.id):
                print("Excluding {}.".format(seance.user.username))
                invalid_users.append(seance.user_id)

        return [x.id for x in seances if x.user_id not in invalid_users]

    def is_valid_seance(
        self, seance_id, sensors=["acc", "gyro", "fsr", "cpu", "mem", "net"]
    ):
        """
        Check if the seance is valid.
        """
        print()
        print("Checking seance {}.".format(seance_id))
        seance = Seance.objects.get(id=seance_id)

        # Seances shorter than a minute are automatically invalid
        if seance.end - seance.start < timedelta(seconds=60 * 3):
            print(
                "Removing seance {}, because of insufficient length.".format(seance_id)
            )
            return False

        # If only one seance per user per task, those are invalid as well
        if (
            Seance.objects.filter(
                experiment__sequence_number=self.task_number, user=seance.user
            ).count()
            < 2
        ):
            print(
                "Removing seance {}, because of a lack of a partner seance.".format(
                    seance_id
                )
            )
            return False

        query = Q(topic__contains=sensors[0])
        for sensor in sensors[1:]:
            query |= Q(topic__contains=sensor)

        sensor_ids = [x.id for x in Sensor.objects.filter(query)]

        for sensor_id in sensor_ids:
            if not self._is_valid_signal(seance_id, sensor_id):
                return False
        return True

    @staticmethod
    def _is_valid_signal(seance_id, sensor_id):
        """
        Check if the signal is valid.
        """
        records = SensorRecord.objects.filter(
            seance_id=seance_id, sensor_id=sensor_id
        ).order_by("timestamp")

        # Check if there is data for sensor, if not, exclude.
        if not records.count():
            print(
                "Removing seance {}, because of no data for {}.".format(
                    seance_id, Sensor.objects.get(id=sensor_id).topic
                )
            )
            return False

        return True

    def _get_seance_ids(self):
        """
        Get seance ids with the given task number.
        """
        return [
            x.id
            for x in Seance.objects.filter(
                valid=True, experiment__sequence_number=self.task_number
            ).order_by("created")
        ]


class CsvDataHandler:
    def __init__(
        self,
        data_dir="/home/oper/Projects/cabackend/jupyter/data/",
        file_name_template="segmented_data_{}_seconds_experiment_{}.csv",
    ):
        self.data_dir = data_dir
        self.file_name_template = file_name_template

    # GENERATING CSV DATA
    def generate_csv_data(self, seance_ids, task=1, seconds=1):
        """
        Generate csv file with calculated features, from data subsampled to the given time interval.
        """
        step = timedelta(seconds=seconds)
        seances = Seance.objects.filter(id__in=seance_ids).order_by("start")

        print(
            "Generating segmented data csv file with {} seconds intervals for {} seances.".format(
                seconds, seances.count()
            )
        )

        file_name = self.data_dir + self.file_name_template.format(seconds, task)
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
            "ir_sd": [],
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
                list(self._load_data(seance.id, "accelerometer"))
                + list(self._load_data(seance.id, "gyroscope"))
                + list(self._load_data(seance.id, "force"))
                + list(self._load_data(seance.id, "cpu"))
                + [self._load_data(seance.id, "ram")]
                + list(self._load_data(seance.id, "net"))
            )
            pir_data = self._load_data(seance.id, "pir")

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
                ax_val, _, _, ax_me, ax_sd = self._process_signal(sub_data[0])
                ay_val, _, _, ay_me, ay_sd = self._process_signal(sub_data[1])
                az_val, _, _, az_me, az_sd = self._process_signal(sub_data[2])
                a_val, a_me, a_sd = self._join_accelerometer_signals(
                    ax_val, ay_val, az_val
                )
                ax_mcr = self._mean_crossing_rate(ax_val, ax_me)
                ay_mcr = self._mean_crossing_rate(ay_val, ay_me)
                az_mcr = self._mean_crossing_rate(az_val, az_me)
                a_mcr = self._mean_crossing_rate(a_val, a_me)
                ax_mai = self._mean_acceleration_intensity(ax_val)
                ay_mai = self._mean_acceleration_intensity(ay_val)
                az_mai = self._mean_acceleration_intensity(az_val)
                a_mai = self._mean_acceleration_intensity(a_val)
                ax_bin = self._get_10_mag_bins(sub_data[0])
                ay_bin = self._get_10_mag_bins(sub_data[1])
                az_bin = self._get_10_mag_bins(sub_data[2])

                # gyroscope
                gx_val, _, _, gx_me, gx_sd = self._process_signal(sub_data[3])
                gy_val, _, _, gy_me, gy_sd = self._process_signal(sub_data[4])
                gz_val, _, _, gz_me, gz_sd = self._process_signal(sub_data[5])
                g_val, g_me, g_sd = self._join_accelerometer_signals(
                    gx_val, gy_val, gz_val
                )
                gx_mcr = self._mean_crossing_rate(gx_val, gx_me)
                gy_mcr = self._mean_crossing_rate(gy_val, gy_me)
                gz_mcr = self._mean_crossing_rate(gz_val, gz_me)
                g_mcr = self._mean_crossing_rate(g_val, g_me)
                gx_bin = self._get_10_mag_bins(sub_data[3])
                gy_bin = self._get_10_mag_bins(sub_data[4])
                gz_bin = self._get_10_mag_bins(sub_data[5])

                # force
                fa_val, _, _, fa_me, fa_sd = self._process_signal(sub_data[6])
                fb_val, _, _, fb_me, fb_sd = self._process_signal(sub_data[7])
                fc_val, _, _, fc_me, fc_sd = self._process_signal(sub_data[8])
                fd_val, _, _, fd_me, fd_sd = self._process_signal(sub_data[9])
                fa_mcr = self._mean_crossing_rate(fa_val, fa_me)
                fb_mcr = self._mean_crossing_rate(fb_val, fb_me)
                fc_mcr = self._mean_crossing_rate(fc_val, fc_me)
                fd_mcr = self._mean_crossing_rate(fd_val, fd_me)
                fa_bin = self._get_10_mag_bins(sub_data[6])
                fb_bin = self._get_10_mag_bins(sub_data[7])
                fc_bin = self._get_10_mag_bins(sub_data[8])
                fd_bin = self._get_10_mag_bins(sub_data[9])

                # cpu
                ca_val, _, _, ca_me, ca_sd = self._process_signal(sub_data[10])
                cb_val, _, _, cb_me, cb_sd = self._process_signal(sub_data[11])
                cc_val, _, _, cc_me, cc_sd = self._process_signal(sub_data[12])
                cd_val, _, _, cd_me, cd_sd = self._process_signal(sub_data[13])
                c_val, c_me, c_sd = self._join_cpu_signals(
                    ca_val, cb_val, cc_val, cd_val
                )
                ca_min, ca_max, ca_mcr = self._get_cpu_stats(ca_val)
                cb_min, cb_max, cb_mcr = self._get_cpu_stats(cb_val)
                cc_min, cc_max, cc_mcr = self._get_cpu_stats(cc_val)
                cd_min, cd_max, cd_mcr = self._get_cpu_stats(cd_val)
                c_min, c_max, c_mcr = self._get_cpu_stats(c_val)

                # ram
                m_val, _, _, m_me, m_sd = self._process_signal(sub_data[14])
                derivatives, peaks = self._find_ram_jump(m_val)
                m_me, m_min, m_max, m_jc, m_jr, m_jv, m_iji = self._get_mem_stats(
                    m_val, peaks, derivatives
                )

                # net
                ns_val, _, _, ns_me, ns_sd = self._process_signal(sub_data[15])
                nr_val, _, _, nr_me, nr_sd = self._process_signal(sub_data[16])
                ns_der, ns_pe = self._find_net_jump(ns_val)
                nr_der, nr_pe = self._find_net_jump(nr_val)
                ns_sum, ns_jc, ns_jr, ns_jv, ns_iji = self._get_net_stats(
                    ns_val, ns_pe, ns_der
                )
                nr_sum, nr_jc, nr_jr, nr_jv, nr_iji = self._get_net_stats(
                    nr_val, nr_pe, nr_der
                )

                s = datetime.now()
                # ir sensors
                pir_datas = self._process_pir_data(
                    pir_data, start, start + step, seconds, interval=1
                )
                ir_me = pir_datas[0][0]
                ir_sd = pir_datas[1][0]

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
                resulting_data["ir_sd"].append(ir_sd)
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
                if i % 15 == 0:
                    print(i)
        df = DataFrame(resulting_data)
        df.to_csv(file_name, index=False)

    def _get_10_mag_bins(self, topic_records):
        """
        Calculate 10 magnitude bins, based on frequencies
        """
        if not len(topic_records):
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        values = [x.value for x in topic_records]
        tim = [x.timestamp for x in topic_records]
        step = 5
        values_inter, tim_inter = self._interpolate(topic_records, step)
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

    @staticmethod
    def _interpolate(data, step: int):
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

    @staticmethod
    def _process_pir_data(records, start, end, segment_interval, interval=1):
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

    @staticmethod
    def _load_data(seance_id, sens):
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
            return SensorRecord.objects.filter(
                seance=seance, sensor=sensors[0]
            ).order_by("timestamp")
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
            return SensorRecord.objects.filter(
                seance=seance, sensor__in=sensors
            ).order_by("timestamp")
        else:
            raise ValueError("Invalid sensor string.")

    @staticmethod
    def _process_signal(records):
        """
        Take Django query and do basic signal processing.
        """
        values = [x.value for x in records]
        times = [x.timestamp for x in records]
        m = mean(values)
        s = std(values)
        norm = [(x - m) / s for x in values]

        return values, times, norm, m, s

    @staticmethod
    def _join_accelerometer_signals(x, y, z):
        """
        Join accelerometer signals, based simply on concurrence.
        We can do this, as only one controller sends data in loop for all axis.
        """
        result = []
        n = min(len(x), len(y), len(z))
        for a, b, c in zip(x[:n], y[:n], z[:n]):
            result.append(sqrt(a ** 2 + b ** 2 + c ** 2))
        return result, mean(result), std(result)

    @staticmethod
    def _mean_crossing_rate(signal, m):
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

    @staticmethod
    def _mean_acceleration_intensity(signal):
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

    @staticmethod
    def _join_cpu_signals(a, b, c, d):
        """
        Similar to accelerometer one.
        """
        result = []
        n = min(len(a), len(b), len(c), len(d))
        for w, x, y, z in zip(a[:n], b[:n], c[:n], d[:n]):
            result.append(sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2))
        return result, mean(result), std(result)

    def _get_cpu_stats(self, val):
        if not val:
            return 0, 0, 0
        return min(val), max(val), self._mean_crossing_rate(val, mean(val))

    @staticmethod
    def _find_ram_jump(signal):
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

    @staticmethod
    def _get_mem_stats(val, peaks, derivatives):
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

    @staticmethod
    def _find_net_jump(signal):
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

    @staticmethod
    def _get_net_stats(val, peaks, derivatives):
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

    # LOADING CSV DATA
    def load_csv_data(self, time_intervals=[1, 2, 4, 8, 16, 32, 64, 128]):
        """
        Load data from csv files and generate datasets with given time steps.
        Calculate additional frequency features.
        """
        import numpy as np

        data_1 = read_csv(self.data_dir + self.file_name_template.format(1, 1)).fillna(
            0
        )
        data_2 = read_csv(self.data_dir + self.file_name_template.format(1, 2)).fillna(
            0
        )
        data_3 = read_csv(self.data_dir + self.file_name_template.format(1, 3)).fillna(
            0
        )

        datasets = [data_1, data_2, data_3]
        for i in range(len(datasets)):
            for col in [
                "ax_bin",
                "ay_bin",
                "az_bin",
                "gx_bin",
                "gy_bin",
                "gz_bin",
                "fa_bin",
                "fb_bin",
                "fc_bin",
                "fd_bin",
            ]:
                # Transform string of a list into a list of floats
                bins = array(
                    [
                        [float(y) for y in x.strip("[]").split(",")]
                        for x in datasets[i][col]
                    ]
                ).T
                for j in range(len(bins)):
                    datasets[i]["{}_{}".format(col, j)] = bins[j]
                datasets[i] = datasets[i].drop(col, axis=1)
                datasets[i]["{}_max".format(col)] = np.max(bins, axis=0)
                datasets[i]["{}_min".format(col)] = np.min(bins, axis=0)
                datasets[i]["{}_mean".format(col)] = mean(bins, axis=0)
                datasets[i]["{}_std".format(col)] = std(bins, axis=0)
        datasets = [x.fillna(0) for x in datasets]

        def get_n_rows_f(seconds):
            def take_n_rows(df):
                indices = [x // seconds for x in range(len(df))]
                df.insert(df.shape[1], "grouping", indices)
                df = df.groupby("grouping", as_index=False).mean().reset_index()
                df = df.drop("grouping", axis=1)
                return df

            return take_n_rows

        results = {}
        experiment = 1
        for data in datasets:
            for interval in time_intervals:
                results.update(
                    {
                        "{}_{}".format(
                            str(experiment).zfill(2), str(interval).zfill(2)
                        ): data.groupby("seance", as_index=False)
                        .apply(get_n_rows_f(interval))
                        .reset_index()
                        .drop("level_0", axis=1)
                        .drop("level_1", axis=1)
                        .drop("index", axis=1)
                    }
                )
            experiment += 1

        return results
