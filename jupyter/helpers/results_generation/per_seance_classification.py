from datetime import datetime, timedelta
from random import sample

from numpy import argsort, mean
from pandas import unique
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from jupyter.helpers import get_users_data


class PerSeanceClassification:
    def __init__(
        self,
        data,
        task_time,
        users_sample=7,
        random_lda={
            "solver": ["svd", "lsqr", "eigen"],
            "n_components": [1, 2, 3, 4, 5, 6],
        },
        random_forest={
            # Method of selecting samples for training each tree
            "bootstrap": [True, False],
            # Maximum number of levels in tree
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # Number of features to consider at every split
            "max_features": ["auto", "sqrt", "log2", None],
            # Minimum number of samples required at each leaf node
            "min_samples_leaf": [1, 2, 4],
            # Minimum number of samples required to split a node
            "min_samples_split": [2, 5, 10],
            # Number of trees in random forest
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
        random_ada={
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            "base_estimator": [
                DecisionTreeClassifier(
                    max_depth=x, min_samples_leaf=y, min_samples_split=z, max_features=w
                )
                for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
                for y in [1, 2, 4]
                for z in [2, 5, 10]
                for w in ["auto", "sqrt", "log2", None]
            ],
        },
        random_gradient={
            # Maximum number of levels in tree
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # Number of features to consider at every split
            "max_features": ["auto", "sqrt", "log2"],
            # Minimum number of samples required at each leaf node
            "min_samples_leaf": [1, 2, 4],
            # Minimum number of samples required to split a node
            "min_samples_split": [2, 5, 10],
            # Number of trees in random forest
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
        grid_forest={
            # Method of selecting samples for training each tree
            "bootstrap": [True],
            # Maximum number of levels in tree
            "max_depth": [40, 60, 80, None],
            # Number of features to consider at every split
            "max_features": ["auto", "sqrt", "log2", None],
            # Minimum number of samples required at each leaf node
            "min_samples_leaf": [1, 2],
            # Minimum number of samples required to split a node
            "min_samples_split": [2, 5],
            # Number of trees in random forest
            "n_estimators": [400, 800, 1400],
        },
        grid_ada={
            "n_estimators": [200, 600, 1000, 1400, 1800],
            "base_estimator": [
                DecisionTreeClassifier(
                    max_depth=x, min_samples_leaf=y, min_samples_split=z, max_features=w
                )
                for x in [20]
                for y in [2, 4]
                for z in [5, 10]
                for w in ["auto", "sqrt", "log2"]
            ],
        },
        grid_gradient={
            # Maximum number of levels in tree
            "max_depth": [20, 40, 80, None],
            # Number of features to consider at every split
            "max_features": ["auto", "sqrt", "log2"],
            # Minimum number of samples required at each leaf node
            "min_samples_leaf": [2, 4],
            # Minimum number of samples required to split a node
            "min_samples_split": [2, 5],
            # Number of trees in random forest
            "n_estimators": [400, 800, 1400],
        },
    ):
        self.data = data

        self.users_sample = users_sample
        self.random_lda = random_lda
        self.random_forest = random_forest
        self.random_ada = random_ada
        self.random_gradient = random_gradient
        self.grid_forest = grid_forest
        self.grid_ada = grid_ada
        self.grid_gradient = grid_gradient

    def run(self, max_running_time=timedelta(hours=6)):
        (
            random_params,
            grid_params,
            default_accuracies,
            random_accuracies,
            grid_accuracies,
            lda_accuracies,
            stats,
        ) = self.get_results(
            self.data,
            [RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier],
            [self.random_forest, self.random_ada, self.random_gradient],
            [self.grid_forest, self.grid_ada, self.grid_gradient],
            max_running_time,
        )
        ################### TESTING ONLY ###################
        # default_accuracies = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # random_accuracies = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # grid_accuracies = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # lda_accuracies = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # stats = {"test": True}
        ####################################################

        return (
            'from plotly import graph_objects as go\nimport numpy as np\ndefault_accuracies = {}\nrandom_accuracies = {}\ngrid_accuracies = {}\nlda_accuracies = {}\ngraphs = []\nalgorithms = ["random forest", "adaboost", "gradient boosting"]\nfor i in range(3):\n\tgraphs.append(go.Bar(x=["default", "random search", "grid search"],y=[np.mean(default_accuracies[i]),np.mean(random_accuracies[i]),np.mean(grid_accuracies[i]),],name=algorithms[i],))\nfig = go.Figure(data=graphs)\nfig.add_shape(go.layout.Shape(type="line",xref="paper",yref="y",x0=0,y0=np.mean(lda_accuracies),x1=1,y1=np.mean(lda_accuracies),line=dict(color="Red", width=1,)))\nfig.show()\n\n'.format(
                default_accuracies, random_accuracies, grid_accuracies, lda_accuracies
            ),
            stats,
        )

    def get_results(
        self,
        data,
        models,
        parameters_randoms,
        parameters_grids,
        max_running_time,
        epochs=5000,
        iterations=100,
    ):
        s = datetime.now()
        random_params = [[] for _ in range(len(models))]
        grid_params = [[] for _ in range(len(models))]
        default_accuracies = [[] for _ in range(len(models))]
        random_accuracies = [[] for _ in range(len(models))]
        grid_accuracies = [[] for _ in range(len(models))]
        lda_accuracies = []
        stats = {}
        i = 0
        for i in range(epochs):
            # Check if maximum time achieved
            if datetime.now() - s > max_running_time:
                stats["finsihed_on_time"] = False
                stats["running_time"] = datetime.now() - s
                stats["epochs"] = i
                return (
                    random_params,
                    grid_params,
                    default_accuracies,
                    random_accuracies,
                    grid_accuracies,
                    lda_accuracies,
                    stats,
                )
            # Get the training data
            cols = list(data.columns)
            all_users = list(unique(data["user"]))
            users = sorted(sample(all_users, self.users_sample))

            # Subset users
            sub_data = data[data.user.isin(users)]

            # Split the data
            seances = get_users_data(sub_data)
            train_seances = [x[0] for x in seances.values()]
            test_seances = [x[1] for x in seances.values()]
            training_set = sub_data[sub_data["seance"].isin(train_seances)]
            testing_set = sub_data[sub_data["seance"].isin(test_seances)]
            X = training_set.iloc[:, 2:].values
            y = training_set.iloc[:, 0].values.ravel()

            # Feature selection
            f_cols = list(data.iloc[:, 2:].columns)
            sel = SelectKBest(mutual_info_classif, k=2)
            sel.fit(X, y)
            indices = argsort(sel.scores_)

            feature_set = cols[0:2] + [f_cols[x] for x in indices[-10:]]

            X_train = training_set[feature_set].iloc[:, 2:].values
            y_train = training_set[feature_set].iloc[:, 0].values.ravel()
            X_test = testing_set[feature_set].iloc[:, 2:].values
            y_test = testing_set[feature_set].iloc[:, 0].values.ravel()

            # LDA baseline
            classifier = LinearDiscriminantAnalysis()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            lda_accuracies.append(accuracy_score(y_test, y_pred))

            # Searches of the models
            for j in range(len(models)):
                model = models[j]
                classifier = model()
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                default_accuracies[j].append(accuracy_score(y_test, y_pred))

                # Random search
                classifier = model()
                classifier_random = RandomizedSearchCV(
                    estimator=classifier,
                    param_distributions=parameters_randoms[j],
                    n_iter=iterations,
                    cv=3,
                    verbose=0,
                    n_jobs=16,
                )
                classifier_random.fit(X_train, y_train)
                random_params[j].append(classifier_random.best_params_)
                y_pred = classifier_random.best_estimator_.predict(X_test)
                random_accuracies[j].append(accuracy_score(y_test, y_pred))

                # Grid search
                classifier = model()
                classifier_grid = GridSearchCV(
                    estimator=classifier,
                    param_grid=parameters_grids[j],
                    cv=3,
                    n_jobs=16,
                    verbose=0,
                )
                classifier_grid.fit(X_train, y_train)
                grid_params[j].append(classifier_grid.best_params_)
                y_pred = classifier_grid.best_estimator_.predict(X_test)
                grid_accuracies[j].append(accuracy_score(y_test, y_pred))

        stats["finsihed_on_time"] = False
        stats["running_time"] = datetime.now() - s
        stats["epochs"] = i + 1
        return (
            random_params,
            grid_params,
            default_accuracies,
            random_accuracies,
            grid_accuracies,
            lda_accuracies,
            stats,
        )

    def __str__(self):
        return "Per seance classification approach"
