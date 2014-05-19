import hyperopt.pyll as pyll


class NoModelException(Exception):
    def __init__(self, cls, method):
        self.cls = cls
        self.method = method

    def __str__(self):
        return repr("You called %s.%s without specifying a model first."
                    % (type(self.cls), self.method))


def hp_pchoice(label, p_options):
    """
    label: string
    p_options: list of (probability, option) pairs
    """
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    p, options = zip(*p_options)
    n_options = len(options)
    ch = pyll.scope.hyperopt_param(label,
                              pyll.scope.categorical(
                                  p,
                                  upper=n_options))
    return pyll.scope.switch(ch, *options)


def hp_choice(label, options):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    ch = pyll.scope.hyperopt_param(label,
        pyll.scope.randint(len(options)))
    return pyll.scope.switch(ch, *options)


def hp_randint(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
        pyll.scope.randint(*args, **kwargs))


def hp_uniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.uniform(*args, **kwargs)))


def hp_quniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.quniform(*args, **kwargs)))


def hp_loguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.loguniform(*args, **kwargs)))


def hp_qloguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.qloguniform(*args, **kwargs)))


def hp_normal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.normal(*args, **kwargs)))


def hp_qnormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.qnormal(*args, **kwargs)))


def hp_lognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.lognormal(*args, **kwargs)))


def hp_qlognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.float(
            pyll.scope.hyperopt_param(label,
                pyll.scope.qlognormal(*args, **kwargs)))