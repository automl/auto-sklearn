from .classification import ParamSklearnClassifier


class ParamSklearnTextClassifier(ParamSklearnClassifier):
    @classmethod
    def get_hyperparameter_search_space(cls, include_estimators=None,
                                        exclude_estimators=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        dataset_properties=None):
        if include_preprocessors is None:
            if exclude_preprocessors is None:
                exclude_preprocessors = ["rescaling"]
            elif isinstance(exclude_preprocessors, list):
                exclude_preprocessors.append(exclude_preprocessors)
            else:
                raise TypeError()

        # @Stefan: you can exclude classifiers and preprocessing methods here
        # From here: http://blog.devzero.com/2013/01/28/how-to-override-a-class-method-in-python/
        cs = super(ParamSklearnTextClassifier, cls).\
            get_hyperparameter_search_space(
            include_estimators=include_estimators,
            exclude_estimators=exclude_estimators,
            include_preprocessors=include_preprocessors,
            exclude_preprocessors=exclude_preprocessors,
            dataset_properties=dataset_properties
        )

        return cs

    @staticmethod
    def _get_pipeline():
        # TODO @Stefan: you probably want to add row normalization after the
        # preprocessing step
        return ["imputation", "__preprocessor__", "__estimator__"]