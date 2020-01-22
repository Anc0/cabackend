from datetime import datetime, timedelta
from random import sample

from numpy import argsort, mean
from pandas import unique, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from jupyter.helpers import get_users_data


class PerSeanceClassification:
    def __init__(
        self,
        data,
        task_time,
        users_sample=7,
        params=[
            [
                {
                    "n_estimators": 1800,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": None,
                    "max_depth": 80,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 2000,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 90,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 800,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 30,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": None,
                    "max_depth": 70,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": None,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 50,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 20,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 80,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 5,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "max_depth": 60,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1600,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "max_depth": 80,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": None,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 2000,
                    "min_samples_split": 10,
                    "min_samples_leaf": 2,
                    "max_features": None,
                    "max_depth": 50,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 60,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "max_depth": 20,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 100,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 60,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": 60,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1000,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": None,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 30,
                    "bootstrap": True,
                },
            ],
            [
                {
                    "n_estimators": 600,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 20,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 10,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 40,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 90,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 20,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1600,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 90,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 20,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 600,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "auto",
                    "max_depth": 80,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 5,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 70,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 5,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 60,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "auto",
                    "max_depth": 30,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1000,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 50,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 90,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 800,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "max_depth": 20,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": 20,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "max_depth": 100,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": 80,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 800,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 20,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1000,
                    "min_samples_split": 10,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 40,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": None,
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1000,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "max_depth": 90,
                    "bootstrap": False,
                },
            ],
            [
                {
                    "n_estimators": 1800,
                    "min_samples_split": 10,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "max_depth": 50,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1400,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 100,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": 90,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1000,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": None,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 2000,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": None,
                    "max_depth": 10,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 5,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "max_depth": 20,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 1600,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 30,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "auto",
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                    "max_depth": 80,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 70,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1800,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "auto",
                    "max_depth": 90,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1600,
                    "min_samples_split": 2,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "max_depth": 50,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 10,
                    "min_samples_leaf": 2,
                    "max_features": "auto",
                    "max_depth": 80,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 400,
                    "min_samples_split": 5,
                    "min_samples_leaf": 1,
                    "max_features": "log2",
                    "max_depth": 40,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 5,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 40,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 800,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "max_depth": 60,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "min_samples_leaf": 4,
                    "max_features": "auto",
                    "max_depth": 30,
                    "bootstrap": True,
                },
                {
                    "n_estimators": 200,
                    "min_samples_split": 10,
                    "min_samples_leaf": 2,
                    "max_features": None,
                    "max_depth": 100,
                    "bootstrap": False,
                },
                {
                    "n_estimators": 1600,
                    "min_samples_split": 10,
                    "min_samples_leaf": 2,
                    "max_features": "auto",
                    "max_depth": 90,
                    "bootstrap": True,
                },
            ],
        ],
    ):
        self.data = data
        self.task_time = task_time

        self.task = int(task_time.split("_")[0])

        self.users_sample = users_sample
        self.params = params

    def run(self, max_running_time=timedelta(hours=6)):
        keys, scores, stats = self.get_results(self.data, max_running_time)
        ################### TESTING ONLY ###################
        ####################################################
        return (
            'from plotly import graph_objects as go\nfig = go.Figure(data=[go.Bar(x={}, y={}, name="Task 1")])\nfig.upd'
            'ate_layout(xaxis_type="category")\nfig.show()'.format(keys, scores),
            stats,
        )

    def get_results(
        self, data: DataFrame, max_running_time, epochs=5000, iterations=100,
    ):
        s = datetime.now()
        results = {}
        stats = {}
        finished = True
        e = 0
        for e in range(epochs):
            # Check if maximum time achieved
            if datetime.now() - s > max_running_time:
                finished = False
                e -= 1
                break

            # Subset users
            all_users = list(unique(data["user"]))
            users = sorted(sample(all_users, self.users_sample))
            sub_data = data[data.user.isin(users)]

            # Train all models
            models, sub_data, u = self.get_models(
                RandomForestClassifier,
                users,
                sub_data,
                params=self.params[self.task - 1],
                iterations=iterations,
            )

            # Retrieve testing data
            X_test = sub_data[2]
            y_test = sub_data[3]

            # Predict all test samples with all models
            probabilities = []
            for model in models:
                probabilities.append(model.predict_proba(X_test))

            # Probability prediction
            probs = self.get_probs_per_test_case_single_task(
                users, probabilities, y_test
            )
            for i, user in enumerate(users):
                key = str(user).zfill(2)
                if key in results:
                    results[key].append(probs[i])
                else:
                    results.update({key: [probs[i]]})

        keys = []
        scores = []
        for key in results:
            keys.append(key)
            scores.append(mean(results[key]))

        order = argsort(keys)
        keys = [keys[x] for x in order]
        scores = [scores[x] for x in order]

        stats["finsihed_on_time"] = finished
        stats["running_time"] = datetime.now() - s
        stats["epochs"] = e + 1

        return keys, scores, stats

    def get_models(
        self,
        model,
        users,
        data,
        params=[],
        search=False,
        search_params={},
        iterations=100,
    ):

        cols = list(data.columns)

        # Split the data
        seances = get_users_data(data)
        train_seances = [x[0] for x in seances.values()]
        test_seances = [x[1] for x in seances.values()]
        training_set = data[data["seance"].isin(train_seances)]
        testing_set = data[data["seance"].isin(test_seances)]
        X = training_set.iloc[:, 2:].values
        y = training_set.iloc[:, 0].values.ravel()

        # Feature selection on training data
        f_cols = list(data.iloc[:, 2:].columns)
        sel = SelectKBest(mutual_info_classif, k=2)
        sel.fit(X, y)
        indices = argsort(sel.scores_)
        feature_set = cols[0:2] + [f_cols[x] for x in indices[-10:]]

        # Train len(users) classifiers
        X_train = training_set[feature_set].iloc[:, 2:].values
        y_train = training_set[feature_set].iloc[:, 0].values.ravel()
        X_test = testing_set[feature_set].iloc[:, 2:].values
        y_test = testing_set[feature_set].iloc[:, 0].values.ravel()
        models = []

        for i, user in enumerate(users):
            y_mod = [1 if x == user else 0 for x in y_train]
            if search:
                classf = model()
                classifier = RandomizedSearchCV(
                    estimator=classf,
                    param_distributions=search_params,
                    n_iter=iterations,
                    cv=3,
                    verbose=0,
                    n_jobs=16,
                )
                classifier.fit(X_train, y_mod)
                models.append(classifier.best_estimator_)
            else:
                classifier = model(**params[i])
                classifier.fit(X_train, y_mod)
                models.append(classifier)
        return models, (X_train, y_train, X_test, y_test), users

    @staticmethod
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
            preds.append(accuracy_score(y_tmp_test, y_tmp_pred))
        return preds

    def __str__(self):
        return "Multi binary classifiers approach"
