import sklearn.utils

def get_random_metric(random_state=1):
    random_state = sklearn.utils.check_random_state(random_state)

    def _random(d1, d2):
        return random_state.random_sample()

    return _random