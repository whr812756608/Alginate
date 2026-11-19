import numpy as np

NOMINAL = (0, 20, 40, 50, 60, 80, 100)

def composition_loo(nominal_m_pct):
    a = np.asarray(nominal_m_pct)
    if a.ndim != 1 or sorted(set(a.tolist())) != list(NOMINAL):
        raise ValueError("Expected seven nominal calibration compositions")
    for held in NOMINAL:
        train = np.flatnonzero(a != held)
        test = np.flatnonzero(a == held)
        if not len(train) or not len(test):
            raise ValueError("Empty train/test group")
        yield held, train, test
