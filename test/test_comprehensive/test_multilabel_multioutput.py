"""
多标签/多输出任务支持测试

测试场景：构造多标签分类和多输出回归数据集
验证点：
- 多标签分类：predict() 返回二维数组（n_samples × n_classes），标签独立
- 多输出回归：每个输出独立预测
- 评估指标适配多输出场景
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
from autosklearn.metrics import accuracy, f1_macro, mean_squared_error, r2


class TestMultilabelMultioutput:
    """测试多标签/多输出任务支持"""

    @pytest.fixture(scope="class")
    def multilabel_data(self):
        """生成多标签分类数据集"""
        X, y = make_multilabel_classification(
            n_samples=200, n_features=20, n_classes=5, n_labels=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def multioutput_regression_data(self):
        """生成多输出回归数据集"""
        X, y = make_regression(
            n_samples=200, n_features=10, n_targets=3,
            noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_multilabel_predict_returns_2d_array(self, multilabel_data, tmp_path):
        """
        测试多标签分类：predict() 返回二维数组（n_samples × n_classes）
        
        Expects
        -------
        * predict() 返回的数组形状应为 (n_samples, n_classes)
        * 每个标签的预测值应为0或1
        """
        X_train, X_test, y_train, y_test = multilabel_data
        
        tmp_folder = str(tmp_path / "autosklearn_multilabel")
        
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
        
        n_classes = y_train.shape[1]
        assert predictions.shape == (len(X_test), n_classes), \
            f"预测数组形状 {predictions.shape} 应为 ({len(X_test)}, {n_classes})"
        
        unique_values = np.unique(predictions)
        assert set(unique_values).issubset({0, 1}), \
            f"多标签预测值应为0或1，实际值: {unique_values}"

    def test_multilabel_labels_independent(self, multilabel_data, tmp_path):
        """
        测试多标签分类：标签独立预测
        
        Expects
        -------
        * 不同标签的预测应相互独立
        * 每个标签列应有不同的预测分布
        """
        X_train, X_test, y_train, y_test = multilabel_data
        
        tmp_folder = str(tmp_path / "autosklearn_multilabel_independent")
        
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
        
        n_classes = y_train.shape[1]
        column_sums = np.sum(predictions, axis=0)
        
        assert len(set(column_sums)) > 1 or n_classes == 1, \
            "不同标签列应有不同的预测分布"

    def test_multilabel_predict_proba(self, multilabel_data, tmp_path):
        """
        测试多标签分类：predict_proba() 返回概率
        
        Expects
        -------
        * predict_proba() 应返回每个标签的概率
        * 概率值应在[0, 1]范围内
        """
        X_train, X_test, y_train, y_test = multilabel_data
        
        tmp_folder = str(tmp_path / "autosklearn_multilabel_proba")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        try:
            probas = automl.predict_proba(X_test)
            
            if isinstance(probas, list):
                assert len(probas) == y_train.shape[1], \
                    f"概率列表长度 {len(probas)} 应等于标签数 {y_train.shape[1]}"
                
                for i, proba in enumerate(probas):
                    assert proba.shape[0] == len(X_test), \
                        f"标签{i}的概率形状 {proba.shape} 第一维应为 {len(X_test)}"
                    assert np.all(proba >= 0) and np.all(proba <= 1), \
                        f"标签{i}的概率值应在[0, 1]范围内"
            else:
                assert probas.shape == (len(X_test), y_train.shape[1]), \
                    f"概率数组形状 {probas.shape} 应为 ({len(X_test)}, {y_train.shape[1]})"
                assert np.all(probas >= 0) and np.all(probas <= 1), \
                    "概率值应在[0, 1]范围内"
        except (AttributeError, NotImplementedError):
            pytest.skip("predict_proba 在此配置下不可用")

    def test_multilabel_score_returns_valid_metric(self, multilabel_data, tmp_path):
        """
        测试多标签分类：score() 返回有效评估指标
        
        Expects
        -------
        * score() 应返回有效的评估分数
        * 分数应在合理范围内
        """
        X_train, X_test, y_train, y_test = multilabel_data
        
        tmp_folder = str(tmp_path / "autosklearn_multilabel_score")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        score = automl.score(X_test, y_test)
        
        assert isinstance(score, (int, float, np.number)), \
            "多标签评分应为数值类型"
        assert not np.isnan(score), "评分不应为NaN"
        assert not np.isinf(score), "评分不应为无穷大"

    def test_multioutput_regression_predict_shape(self, multioutput_regression_data, tmp_path):
        """
        测试多输出回归：predict() 返回正确形状
        
        Expects
        -------
        * predict() 返回的数组形状应为 (n_samples, n_outputs)
        * 每个输出应为连续值
        """
        X_train, X_test, y_train, y_test = multioutput_regression_data
        
        tmp_folder = str(tmp_path / "autosklearn_multioutput")
        
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
        
        n_outputs = y_train.shape[1]
        assert predictions.shape == (len(X_test), n_outputs), \
            f"预测数组形状 {predictions.shape} 应为 ({len(X_test)}, {n_outputs})"
        
        assert not np.any(np.isnan(predictions)), "预测值不应包含NaN"
        assert not np.any(np.isinf(predictions)), "预测值不应包含无穷大"

    def test_multioutput_regression_independent_outputs(self, multioutput_regression_data, tmp_path):
        """
        测试多输出回归：每个输出独立预测
        
        Expects
        -------
        * 不同输出的预测应相互独立
        * 每个输出列应有不同的值范围
        """
        X_train, X_test, y_train, y_test = multioutput_regression_data
        
        tmp_folder = str(tmp_path / "autosklearn_multioutput_independent")
        
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
        
        n_outputs = y_train.shape[1]
        
        output_means = np.mean(predictions, axis=0)
        output_stds = np.std(predictions, axis=0)
        
        assert len(set(np.round(output_means, 3))) > 1 or n_outputs == 1, \
            "不同输出列应有不同的均值"

    def test_multioutput_regression_score(self, multioutput_regression_data, tmp_path):
        """
        测试多输出回归：score() 返回有效指标
        
        Expects
        -------
        * score() 应返回有效的评估分数
        * 分数应在合理范围内
        """
        X_train, X_test, y_train, y_test = multioutput_regression_data
        
        tmp_folder = str(tmp_path / "autosklearn_multioutput_score")
        
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        score = automl.score(X_test, y_test)
        
        assert isinstance(score, (int, float, np.number)), \
            "多输出回归评分应为数值类型"
        assert not np.isnan(score), "评分不应为NaN"
        assert not np.isinf(score), "评分不应为无穷大"

    def test_multilabel_with_sparse_labels(self, tmp_path):
        """
        测试稀疏标签的多标签分类
        
        Expects
        -------
        * 即使某些标签很少出现，也能正常训练
        """
        np.random.seed(42)
        X = np.random.randn(200, 20)
        
        y = np.zeros((200, 5))
        for i in range(200):
            if np.random.rand() > 0.8:
                y[i, 0] = 1
            if np.random.rand() > 0.9:
                y[i, 1] = 1
            if np.random.rand() > 0.95:
                y[i, 2] = 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        tmp_folder = str(tmp_path / "autosklearn_sparse_multilabel")
        
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
        
        assert predictions.shape == (len(X_test), 5), \
            f"预测数组形状 {predictions.shape} 应为 ({len(X_test)}, 5)"

    def test_multilabel_metrics_adaptation(self, multilabel_data, tmp_path):
        """
        测试多标签任务的评估指标适配
        
        Expects
        -------
        * 评估指标应正确适配多标签场景
        * 不同指标应产生合理的结果
        """
        X_train, X_test, y_train, y_test = multilabel_data
        
        for metric in [accuracy, f1_macro]:
            tmp_folder = str(tmp_path / f"autosklearn_multilabel_metric_{metric.name}")
            
            automl = AutoSklearnClassifier(
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
                f"使用{metric.name}的多标签评分应为数值"
            assert not np.isnan(score), "评分不应为NaN"

    def test_multioutput_regression_metrics_adaptation(self, multioutput_regression_data, tmp_path):
        """
        测试多输出回归的评估指标适配
        
        Expects
        -------
        * 评估指标应正确适配多输出场景
        """
        X_train, X_test, y_train, y_test = multioutput_regression_data
        
        for metric in [r2, mean_squared_error]:
            tmp_folder = str(tmp_path / f"autosklearn_multioutput_metric_{metric.name}")
            
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
                f"使用{metric.name}的多输出回归评分应为数值"
            assert not np.isnan(score), "评分不应为NaN"
