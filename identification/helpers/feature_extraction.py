from numpy import mean

# from seances.models import Seance
# from sensors.models import SensorRecord, Sensor
from seances.models import Seance
from sensors.models import Sensor, SensorRecord

SENSORS = [
    "fsr_01",
    "fsr_02",
    "fsr_03",
    "fsr_04",
    "accel_01_x",
    "accel_01_y",
    "accel_01_z",
    "gyro_01_x",
    "gyro_01_y",
    "gyro_01_z",
    "cpuusage_01",
    "cpuusage_02",
    "cpuusage_03",
    "cpuusage_04",
    "mempercentage_01",
    "netpacketssent_01",
    "netpacketsreceived_01",
]


class AutoEncoder:
    def __init__(self):
        self.raw_data = None

    def run(self):
        # TODO: Only a sample - adapt to own data
        from keras.layers import Input, Dense
        from keras.models import Model

        # this is the size of our encoded representations
        encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

        # this is our input placeholder
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation="relu")(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(784, activation="sigmoid")(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")

        from keras.datasets import mnist
        import numpy as np

        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        autoencoder.fit(
            x_train,
            x_train,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_data=(x_test, x_test),
        )

        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = encoder.predict(x_test)
        decoded_imgs = decoder.predict(encoded_imgs)

        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    SENSORS = [
        "fsr_01",
        "fsr_02",
        "fsr_03",
        "fsr_04",
        "accel01_x",
        "accel01_y",
        "accel01_z",
        "gyro01_x",
        "gyro01_y",
        "gyro01_z",
        "cpuusage_01",
        "cpuusage_02",
        "cpuusage_03",
        "cpuusage_04",
        "mempercentage_01",
        "netpacketssent_01",
        "netpacketsreceived_01",
    ]

    def process_data(self):
        """
        Shape our data in a way that is usable by the autoencoder.
        """
        data = []
        seances = Seance.objects.filter(valid=True, experiment__sequence_number=1)
        sensors = Sensor.objects.filter(topic__in=SENSORS)

        for seance in seances:
            print(seance)
            row_data = {"user_id": seance.user.id, "seance_id": seance.id}
            valid = True
            for sensor in sensors:
                try:
                    sensor_data = self.to_n_points(
                        [
                            x.value
                            for x in SensorRecord.objects.filter(
                                seance=seance, sensor=sensor
                            )
                        ],
                        50,
                    )
                except ValueError:
                    print("Missing data in seance... skipping.")
                    valid = False
                    break
                row_data.update({sensor.topic: sensor_data})
            if valid:
                data.append(row_data)
        return data

    def to_n_points(self, data: list, n: int):
        """
        Take the provided list of values and compress it to a length of n elements.
        This is achieved by averaging elements.
        """
        if len(data) < n:
            raise ValueError("Not enough data to compress to {} elements.".format(n))

        step = len(data) / n
        i = 0
        result = []
        for _ in range(0, n):
            row = data[round(i) : round(i + step)]
            result.append(mean(row))
            i += step
        return result

    data = process_data()


if __name__ == "__main__":
    ae = AutoEncoder()
    ae.process_data()
    ae.run()
