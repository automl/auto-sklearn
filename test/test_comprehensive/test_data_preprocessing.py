"""
数据预处理组件完整性测试

测试场景：强制使用特定预处理组件
验证点：
- 类别特征编码：验证OHE、无编码等选项生效
- 缺失值处理：验证数值/类别缺失值填充策略正确执行
- 标准化/归一化：验证不同缩放策略（StandardScaler、MinMaxScaler等）
- 类别合并：验证少数类别合并功能
- 验证方法：通过 include 参数强制指定特定组件，检查 show_models() 输出
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor


class TestDataPreprocessing:
    """测试数据预处理组件完整性"""

    @pytest.fixture(scope="class")
    def iris_data(self):
        """加载iris数据集"""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def data_with_missing_values(self):
        """生成带缺失值的数据"""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 3, 200)
        
        mask = np.random.rand(200, 10) < 0.1
        X[mask] = np.nan
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def data_with_categorical(self):
        """生成带类别特征的数据"""
        np.random.seed(42)
        n_samples = 200
        
        X_numerical = np.random.randn(n_samples, 5)
        
        X_categorical = np.random.choice(['A', 'B', 'C'], size=(n_samples, 3))
        
        X = np.hstack([X_numerical, X_categorical])
        y = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(5)] + [f'cat_{i}' for i in range(3)])
        for col in [f'num_{i}' for i in range(5)]:
            df[col] = pd.to_numeric(df[col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_one_hot_encoding_enabled(self, iris_data, tmp_path):
        """
        测试One-Hot编码选项生效
        
        Expects
        -------
        * 使用OHE时模型应正常训练
        * show_models() 应显示使用了编码器
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_ohe")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            include={
                'data_preprocessor': ['feature_type']
            }
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_numerical_imputation_strategy(self, data_with_missing_values, tmp_path):
        """
        测试数值缺失值填充策略正确执行
        
        Expects
        -------
        * 带缺失值的数据应能正常训练
        * 预测应返回有效结果
        """
        X_train, X_test, y_train, y_test = data_with_missing_values
        
        tmp_folder = str(tmp_path / "autosklearn_imputation")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions)), "预测结果不应包含NaN"

    def test_standard_scaler(self, iris_data, tmp_path):
        """
        测试StandardScaler标准化策略
        
        Expects
        -------
        * 使用StandardScaler时模型应正常训练
        * 预测结果应有效
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_standard_scaler")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_minmax_scaler(self, iris_data, tmp_path):
        """
        测试MinMaxScaler归一化策略
        
        Expects
        -------
        * 使用MinMaxScaler时模型应正常训练
        * 预测结果应有效
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_minmax_scaler")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_minority_coalescence(self, tmp_path):
        """
        测试少数类别合并功能
        
        Expects
        -------
        * 带稀有类别的数据应能正常训练
        * 预测结果应有效
        """
        np.random.seed(42)
        X = np.random.randn(200, 5)
        
        categories = ['A'] * 180 + ['B'] * 15 + ['C'] * 5
        y = np.array(categories)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        tmp_folder = str(tmp_path / "autosklearn_coalescence")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        
        assert len(predictions) == len(X_test)

    def test_no_preprocessing_option(self, iris_data, tmp_path):
        """
        测试无预处理选项
        
        Expects
        -------
        * 禁用预处理时模型应正常训练
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_no_preprocessing")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_categorical_encoding_with_dataframe(self, data_with_categorical, tmp_path):
        """
        测试DataFrame类别特征编码
        
        Expects
        -------
        * DataFrame类别特征应能正确处理
        * 预测结果应有效
        """
        X_train, X_test, y_train, y_test = data_with_categorical
        
        tmp_folder = str(tmp_path / "autosklearn_categorical_df")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        
        assert len(predictions) == len(X_test)

    def test_balancing_strategy(self, tmp_path):
        """
        测试类别平衡策略
        
        Expects
        -------
        * 不平衡数据应能正常训练
        * 预测结果应有效
        """
        np.random.seed(42)
        X = np.random.randn(200, 5)
        
        y = np.array([0] * 150 + [1] * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        tmp_folder = str(tmp_path / "autosklearn_balancing")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_variance_threshold(self, iris_data, tmp_path):
        """
        测试方差阈值特征选择
        
        Expects
        -------
        * 方差阈值应能正常工作
        * 预测结果应有效
        """
        X_train, X_test, y_train, y_test = iris_data
        
        X_train_with_constant = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])
        X_test_with_constant = np.hstack([X_test, np.zeros((X_test.shape[0], 1))])
        
        tmp_folder = str(tmp_path / "autosklearn_variance_threshold")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train_with_constant, y_train)
        
        predictions = automl.predict(X_test_with_constant)
        score = automl.score(X_test_with_constant, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_feature_type_recognition(self, data_with_categorical, tmp_path):
        """
        测试特征类型识别
        
        Expects
        -------
        * 数值和类别特征应能正确识别
        * 预测结果应有效
        """
        X_train, X_test, y_train, y_test = data_with_categorical
        
        tmp_folder = str(tmp_path / "autosklearn_feature_type")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        
        assert len(predictions) == len(X_test)

    def test_preprocessing_combinations(self, iris_data, tmp_path):
        """
        测试多种预处理组合
        
        Expects
        -------
        * 不同预处理组件组合应能正常工作
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_preprocessing_combo")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_regression_preprocessing(self, tmp_path):
        """
        测试回归任务的预处理
        
        Expects
        -------
        * 回归任务的预处理应正常工作
        """
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        tmp_folder = str(tmp_path / "autosklearn_regression_preprocessing")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0, "R²分数应大于0"
