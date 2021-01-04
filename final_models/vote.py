import numpy as np

def vote(models, policy, weights=None, kind='test'):

    if kind == 'test':
        predictions = [model.predict_test(policy=policy) for model in models]
    else:
        predictions = [model.predict_validation(policy=policy) for model in models]
    n_rows = predictions[0].shape[0]

    if weights is None:
        weights = np.ones((len(models),), dtype=float) 

    if policy == 'hard':
        Y = np.zeros([n_rows, len(models)], dtype=int)
        for i in range(len(models)):
            Y[:, i] = predictions[i]
        y = np.zeros(n_rows)
        for i in range(n_rows):
            y[i] = np.argmax(np.bincount(Y[i,:], weights=weights))
        return y

    else:
        Y = np.zeros([n_rows, len(models), 5], dtype=float)
        for i in range(len(models)):
            Y[:, i, :] = predictions[i]
        Y_proba = np.sum(Y * np.array(weights).reshape((1, len(models), 1)), axis=1)
        y = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            y[i] = np.argmax(Y_proba[i, :])
        return y


