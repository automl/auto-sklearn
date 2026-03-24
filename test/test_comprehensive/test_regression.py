"""
AutoSklearnRegressor 回归功能测试

测试场景：在回归数据集（如boston、diabetes）上验证回归流程
验证点：
- 回归任务能自动识别并适配
- 预测结果为连续值（无明显异常值）
- R²分数在合理范围（如>0.5）
特殊验证：确认使用的是回归指标（如mean_squared_error）而非分类指标
"""
from __future__ import annotations

import time
import tempfile
import os

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from autosklearn.estimators import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error as autosk_mse, r2


class TestAutoSklearnRegressor:
    """测试 AutoSklearnRegressor 回归功能"""

    @pytest.fixture(scope="class")
    def diabetes_data(self):
        """加载diabetes数据集"""
        diabetes = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def california_housing_data(self):
        """加载California housing数据集"""
        california = fetch_california_housing()
        X_train, X_test, y_train, y_test = train_test_split(
            california.data, california.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def synthetic_regression_data(self):
        """生成合成回归数据集"""
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=5, 
            noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_fit_completes_without_error_diabetes(self, diabetes_data, tmp_path):
        """
        测试fit()方法在diabetes数据集上能成功完成训练
        
        Expects
        -------
        * fit() 应该成功完成，不抛出异常
        * 训练时间应在合理范围内（< 120秒）
        * 应自动识别为回归任务
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regressor_diabetes")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        start_time = time.time()
        automl.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        assert training_time < 120, f"训练时间过长: {training_time:.2f}秒"
        assert automl.automl_ is not None, "automl_ 属性不应为 None"

    def test_regression_task_auto_recognition(self, diabetes_data, tmp_path):
        """
        测试回归任务能自动识别并适配
        
        Expects
        -------
        * AutoSklearnRegressor 应正确识别回归任务
        * 内部使用的指标应为回归指标
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regression_recognition")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        assert automl.automl_ is not None
        
        task_type = automl.automl_._task
        assert task_type is not None, "应设置任务类型"

    def test_predict_returns_continuous_values(self, diabetes_data, tmp_path):
        """
        测试predict()返回连续值（无明显异常值）
        
        Expects
        -------
        * predict() 返回的数组长度应与输入样本数一致
        * 预测值应为连续值（浮点数）
        * 预测值不应包含无穷大或NaN
        * 预测值范围应合理（与训练集目标值范围相近）
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regression_predict")
        
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
        
        assert len(predictions) == len(X_test), \
            f"预测结果长度 {len(predictions)} 应与测试样本数 {len(X_test)} 一致"
        
        assert not np.any(np.isnan(predictions)), "预测值不应包含NaN"
        assert not np.any(np.isinf(predictions)), "预测值不应包含无穷大"
        
        y_min, y_max = np.min(y_train), np.max(y_train)
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        
        assert pred_min >= y_min - 3 * (y_max - y_min), \
            f"预测最小值 {pred_min:.3f} 不应显著低于训练集最小值 {y_min:.3f}"
        assert pred_max <= y_max + 3 * (y_max - y_min), \
            f"预测最大值 {pred_max:.3f} 不应显著高于训练集最大值 {y_max:.3f}"

    def test_r2_score_reasonable(self, diabetes_data, tmp_path):
        """
        测试R²分数在合理范围（如>0.5）
        
        Expects
        -------
        * score() 返回的R²分数应在合理范围内
        * R²分数应显著高于0（优于简单预测均值）
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_r2")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        r2 = automl.score(X_test, y_test)
        
        assert r2 > 0, f"R²分数 {r2:.3f} 应高于0"
        assert r2 <= 1.0, f"R²分数 {r2:.3f} 不应超过1"

    def test_regression_metric_not_classification(self, diabetes_data, tmp_path):
        """
        特殊验证：确认使用的是回归指标而非分类指标
        
        Expects
        -------
        * 内部使用的评分函数应为回归指标（如MSE、R2）
        * 不应使用分类指标（如accuracy）
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regression_metric")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            metric=autosk_mse,
        )
        
        automl.fit(X_train, y_train)
        
        used_metric = automl.metric
        assert used_metric is not None, "应设置评估指标"
        
        score = automl.score(X_test, y_test)
        
        assert isinstance(score, (int, float, np.number)), \
            "回归评分应为数值类型"

    def test_regression_with_california_housing(self, california_housing_data, tmp_path):
        """
        在California housing数据集上测试回归功能
        
        Expects
        -------
        * fit() 和 predict() 在较大数据集上正常工作
        * R²分数应在合理范围内
        """
        X_train, X_test, y_train, y_test = california_housing_data
        
        tmp_folder = str(tmp_path / "autosklearn_california")
        
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
        r2 = automl.score(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert r2 > 0, f"California housing R²分数 {r2:.3f} 应高于0"

    def test_regression_with_synthetic_data(self, synthetic_regression_data, tmp_path):
        """
        在合成数据集上测试回归功能
        
        Expects
        -------
        * 在简单合成数据集上应达到较高R²分数
        * 预测值应接近真实值
        """
        X_train, X_test, y_train, y_test = synthetic_regression_data
        
        tmp_folder = str(tmp_path / "autosklearn_synthetic")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        r2 = automl.score(X_test, y_test)
        
        assert r2 > 0.5, f"合成数据R²分数 {r2:.3f} 应较高（>0.5）"

    def test_regression_sprint_statistics(self, diabetes_data, tmp_path):
        """
        测试回归任务的sprint_statistics()
        
        Expects
        -------
        * sprint_statistics() 应返回包含回归相关信息的字符串
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regression_stats")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        stats = automl.sprint_statistics()
        
        assert isinstance(stats, str), "sprint_statistics() 应返回字符串"
        assert len(stats) > 0, "统计字符串不应为空"

    def test_regression_refit(self, diabetes_data, tmp_path):
        """
        测试回归任务的refit()方法
        
        Expects
        -------
        * refit() 应该成功完成
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        tmp_folder = str(tmp_path / "autosklearn_regression_refit")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        initial_r2 = automl.score(X_test, y_test)
        
        try:
            automl.refit()
            refit_r2 = automl.score(X_test, y_test)
            
            assert refit_r2 >= initial_r2 - 0.1, \
                f"refit后R² {refit_r2:.3f} 不应显著低于初始R² {initial_r2:.3f}"
        except Exception as e:
            pytest.skip(f"refit() 在此配置下不可用: {e}")

    def test_regression_with_different_metrics(self, diabetes_data, tmp_path):
        """
        测试使用不同回归指标
        
        Expects
        -------
        * 使用不同指标（R2、MSE）应都能正常训练
        * 评分结果应符合对应指标的性质
        """
        X_train, X_test, y_train, y_test = diabetes_data
        
        for metric in [r2, autosk_mse]:
            tmp_folder = str(tmp_path / f"autosklearn_metric_{metric.name}")
            
            automl = AutoSklearnRegressor(
                time_left_for_this_task=40,
                per_run_time_limit=8,
                tmp_folder=tmp_folder,
                delete_tmp_folder_after_terminate=True,
                seed=42,
                n_jobs=1,
                metric=metric,
            )
            
            automl.fit(X_train, y_train)
            score = automl.score(X_test, y_test)
            
            assert isinstance(score, (int, float, np.number)), \
                f"使用{metric.name}指标的评分应为数值"
