from datetime import timedelta, datetime


class PerSeanceClassification:
    def __init__(
        self, data, task_time, users_sample=7,
    ):
        self.data = data
        self.task_time = task_time

        self.users_sample = users_sample

    def run(self, max_running_time=timedelta(hours=6)):
        """
        Function that gets called by the results generation framework.
        """
        stats = self.get_results(max_running_time,)

        return stats

    def get_results(self, max_running_time):
        """
        Calculate results for given parameters and return them.
        """
        s = datetime.now()
        stats = {}

        epochs = 5000
        s = datetime.now()
        dataset = {
            "users_num": [],
            "features_num": [],
            "segment_interval_length": [],
            "accuracy": [],
            "norm_accuracy": [],
        }
        for users_num in [5, 7, 13]:
            for segment_interval_length in [10, 45, 90, 180]:
                for features_num in [5]:
                    scores = []
                    norms = []
                    for e in range(epochs):
                        if e % 100 == 0:
                            print(e)
                        # Initialization
                        datas = datas_2
                        users = get_random_users(n=users_num, remove=[14], stats=False)

                        data = datas[interval]
                        data = data[data.user.isin(users)]

                        # Split train test
                        seances = get_users_data(data)
                        train_seances = [x[0] for x in seances.values()]
                        test_seances = [x[1] for x in seances.values()]
                        training_set = data[data["seance"].isin(train_seances)]
                        testing_set = data[data["seance"].isin(test_seances)]

                        # Feature selection
                        X = training_set.iloc[:, 2:]
                        y = training_set.iloc[:, 0]
                        f_cols = list(data.iloc[:, 2:].columns)
                        sel = SelectKBest(mutual_info_classif, k=2)
                        sel.fit(X, y)
                        indices = np.argsort(sel.scores_)
                        feature_set = ["user", "seance"] + [
                            f_cols[x] for x in indices[-features_num:]
                        ]
                        training_set = training_set[feature_set]
                        testing_set = testing_set[feature_set]

                        # Calculate distances
                        users = list(pd.unique(training_set["user"]))
                        data = {
                            "train_user": [],
                            "test_user": [],
                            "distance": [],
                            "normalized": [],
                        }
                        for train_user in users:
                            for test_user in users:
                                x = training_set[
                                    training_set["user"] == train_user
                                ].iloc[:, 2:]
                                y = testing_set[testing_set["user"] == test_user].iloc[
                                    :, 2:
                                ]
                                # Distance calculation
                                distance, path = fastdtw(x, y, dist=euclidean)
                                data["train_user"].append("user_" + str(train_user))
                                data["test_user"].append("user_" + str(test_user))
                                data["distance"].append(distance)
                                data["normalized"].append(False)
                                # Normalize signals
                                xcols = list(x.columns)
                                for xcol in xcols:
                                    x[xcol] = [
                                        (a - mean(x[xcol])) / std(x[xcol])
                                        for a in x[xcol]
                                    ]
                                    y[xcol] = [
                                        (a - mean(y[xcol])) / std(y[xcol])
                                        for a in y[xcol]
                                    ]
                                x = x.fillna(0)
                                y = y.fillna(0)
                                # Distance calculation
                                distance, path = fastdtw(x, y, dist=euclidean)
                                data["train_user"].append("user_" + str(train_user))
                                data["test_user"].append("user_" + str(test_user))
                                data["distance"].append(distance)
                                data["normalized"].append(True)

                        # Visualization
                        fig_org = go.Figure(
                            data=go.Heatmap(
                                x=data["test_user"],
                                y=data["train_user"],
                                z=data["distance"],
                                colorscale="temps",
                            )
                        )
                        fig_org.update_layout(
                            title="Distances between training and testing trajectories, {} best features".format(
                                features_num
                            ),
                            xaxis_title="testing user",
                            yaxis_title="training user",
                        )
                        #     fig_org.show()
                        df = DataFrame(data)
                        predictions = []
                        normalizations = []
                        for user in users:
                            data = df[
                                (df["test_user"] == "user_" + str(user))
                                & (df["normalized"] == False)
                            ]
                            predictions.append(
                                list(data["distance"]).index(
                                    min(list(data["distance"]))
                                )
                            )

                            data = df[
                                (df["test_user"] == "user_" + str(user))
                                & (df["normalized"] == True)
                            ]
                            normalizations.append(
                                list(data["distance"]).index(
                                    min(list(data["distance"]))
                                )
                            )

                        scores.append(
                            metrics.accuracy_score(list(range(len(users))), predictions)
                        )
                        norms.append(
                            metrics.accuracy_score(
                                list(range(len(users))), normalizations
                            )
                        )

                    dataset["features_num"].append(features_num)
                    dataset["users_num"].append(users_num)
                    dataset["segment_interval_length"].append(segment_interval_length)
                    dataset["accuracy"].append(mean(scores))
                    dataset["norm_accuracy"].append(mean(norms))

        return stats

    def __str__(self):
        return "Per seance classification approach"
