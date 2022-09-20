from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sklearn.compose
from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from scipy import sparse
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
from autosklearn.pipeline.base import (
    DATASET_PROPERTIES_TYPE,
    PIPELINE_DATA_DTYPE,
    BasePipeline,
)
from autosklearn.pipeline.components.base import (
    AutoSklearnChoice,
    AutoSklearnComponent,
    AutoSklearnPreprocessingAlgorithm,
)
from autosklearn.pipeline.components.data_preprocessing.feature_type_categorical import (  # noqa : E501
    CategoricalPreprocessingPipeline,
)
from autosklearn.pipeline.components.data_preprocessing.feature_type_numerical import (
    NumericalPreprocessingPipeline,
)
from autosklearn.pipeline.components.data_preprocessing.feature_type_text import (
    TextPreprocessingPipeline,
)
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class FeatTypeSplit(AutoSklearnPreprocessingAlgorithm):
    """
    This component is used to apply distinct transformations to categorical,
    numerical and text features of a dataset. It is built on top of sklearn's
    ColumnTransformer.
    """

    def __init__(
        self,
        config: Optional[Configuration] = None,
        pipeline: Optional[BasePipeline] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict[str, Any]] = None,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        force_sparse_output: bool = False,
        column_transformer: Optional[sklearn.compose.ColumnTransformer] = None,
    ):

        if pipeline is not None:
            raise ValueError("DataPreprocessor's argument 'pipeline' should be None")

        self.config = config
        self.pipeline = pipeline
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude
        self.random_state = random_state
        self.init_params = init_params
        self.feat_type = feat_type
        self.force_sparse_output = force_sparse_output

        # The pipeline that will be applied to the categorical features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the categorical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the FeatTypeSplit to the
        # pipeline if needed
        self.categ_ppl = CategoricalPreprocessingPipeline(
            feat_type=self.feat_type,
            config=None,
            steps=pipeline,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            random_state=random_state,
            init_params=init_params,
        )
        # The pipeline that will be applied to the numerical features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the numerical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the FeatTypeSplit to the
        # pipeline if needed
        self.numer_ppl = NumericalPreprocessingPipeline(
            feat_type=self.feat_type,
            config=None,
            steps=pipeline,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            random_state=random_state,
            init_params=init_params,
        )

        # The pipeline that will be applied to the text features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the numerical or categorical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the FeatTypeSplit to the
        # pipeline if needed
        self.txt_ppl = TextPreprocessingPipeline(
            feat_type=self.feat_type,
            config=None,
            steps=pipeline,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            random_state=random_state,
            init_params=init_params,
        )

        if self.feat_type is None:
            self._transformers: List[Tuple[str, AutoSklearnComponent]] = [
                ("categorical_transformer", self.categ_ppl),
                ("numerical_transformer", self.numer_ppl),
                ("text_transformer", self.txt_ppl),
            ]
        else:
            self._transformers: List[Tuple[str, AutoSklearnComponent]] = []
            if "categorical" in self.feat_type.values():
                self._transformers.append(("categorical_transformer", self.categ_ppl))
            if "numerical" in self.feat_type.values():
                self._transformers.append(("numerical_transformer", self.numer_ppl))
            if "string" in self.feat_type.values():
                self._transformers.append(("text_transformer", self.txt_ppl))

        if self.config:
            self.set_hyperparameters(
                feat_type=self.feat_type,
                configuration=self.config,
                init_params=init_params,
            )

        self.column_transformer = column_transformer

    def fit(
        self, X: SUPPORTED_FEAT_TYPES, y: Optional[SUPPORTED_TARGET_TYPES] = None
    ) -> "FeatTypeSplit":

        n_feats = X.shape[1]
        if self.feat_type is not None:
            # Make sure that we are not missing any column!
            expected = set(self.feat_type.keys())
            if hasattr(X, "columns"):
                columns = set(X.columns)
            else:
                columns = set(range(n_feats))
            if expected != columns:
                raise ValueError(
                    f"Train data has columns={expected} yet the"
                    f" feat_types are feat={columns}"
                )
            sklearn_transf_spec = []

            categorical_features = [
                key
                for key, value in self.feat_type.items()
                if value.lower() == "categorical"
            ]
            if len(categorical_features) > 0:
                sklearn_transf_spec.append(
                    ("categorical_transformer", self.categ_ppl, categorical_features)
                )

            numerical_features = [
                key
                for key, value in self.feat_type.items()
                if value.lower() == "numerical"
            ]
            if len(numerical_features) > 0:
                sklearn_transf_spec.append(
                    ("numerical_transformer", self.numer_ppl, numerical_features)
                )

            text_features = [
                key
                for key, value in self.feat_type.items()
                if value.lower() == "string"
            ]
            if len(text_features) > 0:
                sklearn_transf_spec.append(
                    ("text_transformer", self.txt_ppl, text_features)
                )
        else:
            # self.feature_type == None assumes numerical case
            sklearn_transf_spec = [
                ("numerical_transformer", self.numer_ppl, [True] * n_feats)
            ]

        # And one last check in case feat type is None
        # And to make sure the final specification has all the columns
        # considered in the column transformer
        total_columns = sum(
            [len(features) for name, ppl, features in sklearn_transf_spec]
        )
        if total_columns != n_feats:
            raise ValueError(
                "Missing columns in the specification of the data validator"
                f" for train data={np.shape(X)} and spec={sklearn_transf_spec}"
            )

        self.sparse_ = sparse.issparse(X) or self.force_sparse_output
        self.column_transformer = sklearn.compose.ColumnTransformer(
            transformers=sklearn_transf_spec,
            sparse_threshold=float(self.sparse_),
        )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X: SUPPORTED_FEAT_TYPES) -> PIPELINE_DATA_DTYPE:
        if self.column_transformer is None:
            raise ValueError(
                "Cannot call transform on a Datapreprocessor that has not"
                "yet been fit. Please check the log files for errors "
                "while trying to fit the model."
            )
        return self.column_transformer.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "FeatTypeSplit",
            "name": "Feature Type Splitter",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            # TODO find out of this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    def set_hyperparameters(
        self,
        feat_type: FEAT_TYPE_TYPE,
        configuration: Configuration,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> "FeatTypeSplit":
        if init_params is not None and "feat_type" in init_params.keys():
            self.feat_type = init_params["feat_type"]

        self.config = configuration

        for transf_name, transf_op in self._transformers:
            sub_configuration_space = transf_op.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties, feat_type=feat_type
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith("%s:" % transf_name):
                    value = configuration[param]
                    new_name = param.replace("%s:" % transf_name, "", 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(
                sub_configuration_space,
                values=sub_config_dict,
                allow_inactive_with_values=True,
            )

            sub_init_params_dict: Optional[Dict[str, Any]] = None
            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith("%s:" % transf_name):
                        value = init_params[param]
                        new_name = param.replace("%s:" % transf_name, "", 1)
                        sub_init_params_dict[new_name] = value

            if isinstance(
                transf_op, (AutoSklearnChoice, AutoSklearnComponent, BasePipeline)
            ):
                transf_op.set_hyperparameters(
                    feat_type=feat_type,
                    configuration=sub_configuration,
                    init_params=sub_init_params_dict,
                )
            else:
                raise NotImplementedError("Not supported yet!")

        return self

    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        self.dataset_properties = dataset_properties
        cs = ConfigurationSpace()
        cs = FeatTypeSplit._get_hyperparameter_search_space_recursevely(
            feat_type=feat_type,
            dataset_properties=dataset_properties,
            cs=cs,
            transformer=self._transformers,
        )
        return cs

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(
        dataset_properties: DATASET_PROPERTIES_TYPE,
        cs: ConfigurationSpace,
        transformer: BaseEstimator,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
    ) -> ConfigurationSpace:
        for st_name, st_operation in transformer:
            if hasattr(st_operation, "get_hyperparameter_search_space"):
                cs.add_configuration_space(
                    st_name,
                    st_operation.get_hyperparameter_search_space(
                        dataset_properties=dataset_properties
                    ),
                )
            else:
                return FeatTypeSplit._get_hyperparameter_search_space_recursevely(
                    dataset_properties, cs, st_operation
                )
        return cs
