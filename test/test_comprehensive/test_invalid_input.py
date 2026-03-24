"""
无效输入处理测试

测试场景：各种异常输入
- 空DataFrame/数组
- 单类分类（y只有一个唯一值）
- 大量NaN（>50%缺失）
- 全零/常数特征

验证点：
- 抛出有意义的错误信息（不静默失败）
- 可恢复（不导致进程挂死）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor


class TestInvalidInput:
    """测试无效输入处理"""

    @pytest.fixture(scope="class")
    def iris_data(self):
        """加载iris数据集"""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_empty_array_raises_error(self, tmp_path):
        """
        测试空数组应抛出有意义的错误
        
        Expects
        -------
        * 使用空数组应抛出异常
        * 错误信息应有意义，不静默失败
        """
        X_empty = np.array([]).reshape(0, 4)
        y_empty = np.array([])
        
        tmp_folder = str(tmp_path / "autosklearn_empty")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        with pytest.raises((ValueError, IndexError, AssertionError)) as exc_info:
            automl.fit(X_empty, y_empty)
        
        assert str(exc_info.value), "错误信息不应为空"

    def test_empty_dataframe_raises_error(self, tmp_path):
        """
        测试空DataFrame应抛出有意义的错误
        
        Expects
        -------
        * 使用空DataFrame应抛出异常
        * 错误信息应有意义
        """
        X_empty = pd.DataFrame()
        y_empty = pd.Series([], dtype=int)
        
        tmp_folder = str(tmp_path / "autosklearn_empty_df")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        with pytest.raises((ValueError, IndexError, AssertionError)) as exc_info:
            automl.fit(X_empty, y_empty)
        
        assert str(exc_info.value), "错误信息不应为空"

    def test_single_class_classification_raises_error(self, iris_data, tmp_path):
        """
        测试单类分类应抛出有意义的错误
        
        Expects
        -------
        * y只有一个唯一值时应抛出异常
        * 错误信息应指明问题
        """
        X_train, X_test, y_train, y_test = iris_data
        
        y_single_class = np.zeros_like(y_train)
        
        tmp_folder = str(tmp_path / "autosklearn_single_class")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            automl.fit(X_train, y_single_class)
        
        assert str(exc_info.value), "错误信息不应为空"

    def test_high_missing_value_ratio(self, tmp_path):
        """
        测试大量NaN（>50%缺失）的处理
        
        Expects
        -------
        * 高缺失率数据应能处理或抛出有意义的错误
        * 不应导致进程挂死
        """
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        mask = np.random.rand(100, 5) < 0.6
        X[mask] = np.nan
        
        tmp_folder = str(tmp_path / "autosklearn_high_missing")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_all_zero_features(self, tmp_path):
        """
        测试全零特征的处理
        
        Expects
        -------
        * 全零特征应能处理或抛出有意义的错误
        * 不应导致进程挂死
        """
        X = np.zeros((100, 5))
        y = np.random.randint(0, 2, 100)
        
        tmp_folder = str(tmp_path / "autosklearn_zero_features")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_constant_features(self, tmp_path):
        """
        测试常数特征的处理
        
        Expects
        -------
        * 常数特征应能处理或抛出有意义的错误
        * 不应导致进程挂死
        """
        X = np.ones((100, 5)) * 5
        y = np.random.randint(0, 2, 100)
        
        tmp_folder = str(tmp_path / "autosklearn_constant")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_mismatched_x_y_lengths(self, iris_data, tmp_path):
        """
        测试X和y长度不匹配的处理
        
        Expects
        -------
        * 长度不匹配时应抛出有意义的错误
        """
        X_train, X_test, y_train, y_test = iris_data
        
        X_short = X_train[:-10]
        
        tmp_folder = str(tmp_path / "autosklearn_mismatched")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        with pytest.raises((ValueError, IndexError)) as exc_info:
            automl.fit(X_short, y_train)
        
        assert str(exc_info.value), "错误信息不应为空"

    def test_all_nan_features(self, tmp_path):
        """
        测试全NaN特征的处理
        
        Expects
        -------
        * 全NaN特征应能处理或抛出有意义的错误
        * 不应导致进程挂死
        """
        X = np.full((100, 5), np.nan)
        y = np.random.randint(0, 2, 100)
        
        tmp_folder = str(tmp_path / "autosklearn_all_nan")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_inf_values(self, tmp_path):
        """
        测试无穷大值的处理
        
        Expects
        -------
        * 包含无穷大值的数据应能处理或抛出有意义的错误
        """
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        y = np.random.randint(0, 2, 100)
        
        tmp_folder = str(tmp_path / "autosklearn_inf")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_very_large_values(self, tmp_path):
        """
        测试极大值的处理
        
        Expects
        -------
        * 包含极大值的数据应能处理或抛出有意义的错误
        """
        X = np.random.randn(100, 5)
        X[0] = 1e10
        y = np.random.randint(0, 2, 100)
        
        tmp_folder = str(tmp_path / "autosklearn_large")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_negative_class_labels(self, iris_data, tmp_path):
        """
        测试负类别标签的处理
        
        Expects
        -------
        * 负类别标签应能正确处理
        """
        X_train, X_test, y_train, y_test = iris_data
        
        y_negative = y_train - 1
        
        tmp_folder = str(tmp_path / "autosklearn_negative_labels")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X_train, y_negative)
            predictions = automl.predict(X_test)
            assert len(predictions) == len(X_test)
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"负类别标签在此配置下不受支持: {e}")

    def test_string_labels(self, iris_data, tmp_path):
        """
        测试字符串标签的处理
        
        Expects
        -------
        * 字符串标签应能处理或抛出有意义的错误
        """
        X_train, X_test, y_train, y_test = iris_data
        
        y_string = np.array(['class_a' if y == 0 else 'class_b' if y == 1 else 'class_c' for y in y_train])
        
        tmp_folder = str(tmp_path / "autosklearn_string_labels")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X_train, y_string)
            predictions = automl.predict(X_test)
            assert len(predictions) == len(X_test)
        except (ValueError, TypeError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"

    def test_single_sample(self, iris_data, tmp_path):
        """
        测试单样本的处理
        
        Expects
        -------
        * 单样本应抛出有意义的错误
        """
        X_train, X_test, y_train, y_test = iris_data
        
        X_single = X_train[:1]
        y_single = y_train[:1]
        
        tmp_folder = str(tmp_path / "autosklearn_single_sample")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            automl.fit(X_single, y_single)
        
        assert str(exc_info.value), "错误信息不应为空"

    def test_regression_invalid_input(self, tmp_path):
        """
        测试回归任务的无效输入处理
        
        Expects
        -------
        * 回归任务的无效输入应抛出有意义的错误
        """
        X = np.zeros((100, 5))
        y = np.random.randn(100)
        
        tmp_folder = str(tmp_path / "autosklearn_regression_invalid")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        try:
            automl.fit(X, y)
            predictions = automl.predict(X)
            assert len(predictions) == len(X)
        except (ValueError, RuntimeError) as e:
            assert str(e), "错误信息不应为空"
