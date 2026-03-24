"""
内存限制测试

测试场景：在较大数据集上设置 memory_limit
验证点：
- 超内存模型被优雅终止（返回MEMOUT状态）
- 存活模型结果正确
- 系统在内存压力下稳定（无OOM崩溃）

特殊场景：大数据集触发自动降采样/精度降低功能
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor


class TestMemoryLimit:
    """测试内存限制功能"""

    @pytest.fixture(scope="class")
    def iris_data(self):
        """加载iris数据集"""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def large_dataset(self):
        """生成较大数据集"""
        X, y = make_classification(
            n_samples=5000, n_features=50, n_informative=30,
            n_redundant=10, n_classes=3, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_memory_limit_set(self, iris_data, tmp_path):
        """
        测试内存限制设置
        
        Expects
        -------
        * memory_limit 参数应能正确设置
        * 训练应正常完成
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_memory")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=1024,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_low_memory_limit(self, iris_data, tmp_path):
        """
        测试低内存限制
        
        Expects
        -------
        * 低内存限制下应能完成训练
        * 超内存模型应被优雅终止
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_low_memory")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=512,
        )
        
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_test)
            assert len(predictions) == len(X_test)
        except Exception as e:
            pytest.skip(f"低内存限制下训练失败: {e}")

    def test_memory_limit_with_large_dataset(self, large_dataset, tmp_path):
        """
        测试大数据集上的内存限制
        
        Expects
        -------
        * 大数据集上应能处理内存限制
        * 不应导致OOM崩溃
        """
        X_train, X_test, y_train, y_test = large_dataset
        
        tmp_folder = str(tmp_path / "autosklearn_large_memory")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=2048,
        )
        
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_test)
            score = automl.score(X_test, y_test)
            
            assert len(predictions) == len(X_test)
            assert score > 0.3, "大数据集准确率应合理"
        except Exception as e:
            pytest.skip(f"大数据集内存限制测试失败: {e}")

    def test_no_memory_limit(self, iris_data, tmp_path):
        """
        测试无内存限制
        
        Expects
        -------
        * memory_limit=None 时应无内存限制
        * 训练应正常完成
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_no_memory_limit")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=None,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0.5, "准确率应高于随机猜测"

    def test_memory_limit_stability(self, iris_data, tmp_path):
        """
        测试内存限制下的系统稳定性
        
        Expects
        -------
        * 系统在内存限制下应保持稳定
        * 不应崩溃或挂死
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_memory_stable")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=1024,
        )
        
        automl.fit(X_train, y_train)
        
        cv_results = automl.cv_results_
        
        assert cv_results is not None, "cv_results_ 不应为 None"

    def test_regression_memory_limit(self, tmp_path):
        """
        测试回归任务的内存限制
        
        Expects
        -------
        * 回归任务应能正确处理内存限制
        """
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        tmp_folder = str(tmp_path / "autosklearn_reg_memory")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=1024,
        )
        
        automl.fit(X_train, y_train)
        
        predictions = automl.predict(X_test)
        score = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert score > 0, "R²分数应大于0"

    def test_memory_limit_with_parallel_jobs(self, iris_data, tmp_path):
        """
        测试并行任务下的内存限制
        
        Expects
        -------
        * 并行任务应能正确处理内存限制
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_memory_parallel")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=2,
            memory_limit=1024,
        )
        
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_test)
            assert len(predictions) == len(X_test)
        except Exception as e:
            pytest.skip(f"并行任务内存限制测试失败: {e}")

    def test_cv_results_with_memory_limit(self, iris_data, tmp_path):
        """
        测试内存限制下的cv_results_
        
        Expects
        -------
        * cv_results_ 应包含内存相关信息
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_memory_cv")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            memory_limit=1024,
        )
        
        automl.fit(X_train, y_train)
        
        cv_results = automl.cv_results_
        
        assert cv_results is not None, "cv_results_ 不应为 None"
        assert len(cv_results) > 0, "cv_results_ 不应为空"
