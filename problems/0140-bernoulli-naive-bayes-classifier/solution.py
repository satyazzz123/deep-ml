import numpy as np


class NaiveBayes:

    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

        self.classes = None
        self.priors = None
        self.theta = None

    def forward(self, X, y):
        # Fit model to binary features X and labels y

        self.classes = np.unique(y)

        # P(C)
        self.priors = {}

        # theta[i, c] = P(x_i = 1 | C = c)
        self.theta = {}

        for c in self.classes:

            # All samples belonging to class c
            X_c = X[y == c]

            # N_C = number of samples in class c
            N_C = len(X_c)

            # N_i,C = number of samples in class c
            # where feature i is present (1)
            N_i_C = np.sum(X_c, axis=0)

            # P(C)
            self.priors[c] = N_C / len(y)

            # Laplace smoothing:
            #
            # theta_i,C = (N_i,C + alpha) / (N_C + 2*alpha)
            #
            self.theta[c] = (
                N_i_C + self.smoothing
            ) / (
                N_C + 2 * self.smoothing
            )

    def predict(self, X):

        predictions = []

        for x in X:

            scores = {}

            for c in self.classes:

                theta = self.theta[c]

                # Bernoulli likelihood:
                #
                # P(X|C) =
                # product [
                #   theta_i,C^x_i *
                #   (1-theta_i,C)^(1-x_i)
                # ]
                likelihood = np.prod(
                    theta ** x *
                    (1 - theta) ** (1 - x)
                )

                # Naive Bayes score:
                #
                # P(C) * P(X|C)
                scores[c] = self.priors[c] * likelihood

            # Pick class with highest score
            predictions.append(
                max(scores, key=scores.get)
            )

        return np.array(predictions)