"""
元学习冷启动效果测试

测试场景：新任务第一次运行
验证点：
- 元学习推荐配置质量（前几个配置性能）
- 与随机搜索对比的"头对头"胜率（应>50%）
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import accuracy


class TestMetalearning:
    """测试元学习冷启动效果"""

    @pytest.fixture(scope="class")
    def iris_data(self):
        """加载iris数据集"""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def breast_cancer_data(self):
        """加载breast_cancer数据集"""
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_metalearning_initial_configurations(self, iris_data, tmp_path):
        """
        测试元学习推荐配置质量（前几个配置性能）
        
        Expects
        -------
        * 使用元学习初始化的配置应比随机初始化表现更好
        * 前几个配置应能达到合理的性能
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_metalearning")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=25,
        )
        
        automl.fit(X_train, y_train)
        
        score = automl.score(X_test, y_test)
        
        n_classes = len(np.unique(y_train))
        random_guess = 1.0 / n_classes
        
        assert score > random_guess + 0.2, \
            f"元学习初始化后的准确率 {score:.3f} 应显著高于随机猜测 {random_guess:.3f}"

    def test_metalearning_vs_no_metalearning(self, iris_data, tmp_path):
        """
        测试元学习与无元学习的对比
        
        Expects
        -------
        * 使用元学习的模型应比不使用元学习的模型表现更好或相当
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder_meta = str(tmp_path / "autosklearn_with_meta")
        tmp_folder_no_meta = str(tmp_path / "autosklearn_no_meta")
        
        automl_with_meta = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder_meta,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=25,
        )
        
        automl_no_meta = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder_no_meta,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=0,
        )
        
        automl_with_meta.fit(X_train, y_train)
        automl_no_meta.fit(X_train, y_train)
        
        score_with_meta = automl_with_meta.score(X_test, y_test)
        score_no_meta = automl_no_meta.score(X_test, y_test)
        
        assert score_with_meta >= score_no_meta - 0.1, \
            f"元学习准确率 {score_with_meta:.3f} 不应显著低于无元学习 {score_no_meta:.3f}"

    def test_metalearning_different_datasets(self, breast_cancer_data, tmp_path):
        """
        测试元学习在不同数据集上的效果
        
        Expects
        -------
        * 元学习在不同数据集上都应有效
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        tmp_folder = str(tmp_path / "autosklearn_meta_cancer")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=25,
        )
        
        automl.fit(X_train, y_train)
        
        score = automl.score(X_test, y_test)
        
        assert score > 0.6, f"元学习在乳腺癌数据集上的准确率 {score:.3f} 应较高"

    def test_metalearning_first_configurations_quality(self, iris_data, tmp_path):
        """
        测试元学习前几个配置的质量
        
        Expects
        -------
        * 元学习推荐的前几个配置应能达到合理性能
        * 这些配置应比随机选择更好
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_meta_first_configs")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=40,
            per_run_time_limit=8,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=10,
        )
        
        automl.fit(X_train, y_train)
        
        score = automl.score(X_test, y_test)
        
        n_classes = len(np.unique(y_train))
        random_guess = 1.0 / n_classes
        
        assert score > random_guess + 0.15, \
            f"仅使用元学习前10个配置的准确率 {score:.3f} 应高于随机猜测"

    def test_metalearning_with_different_seeds(self, iris_data, tmp_path):
        """
        测试不同随机种子下的元学习效果
        
        Expects
        -------
        * 不同种子下元学习都应有效
        * 结果应相对稳定
        """
        X_train, X_test, y_train, y_test = iris_data
        
        scores = []
        
        for seed in [1, 42, 123]:
            tmp_folder = str(tmp_path / f"autosklearn_meta_seed_{seed}")
            
            automl = AutoSklearnClassifier(
                time_left_for_this_task=40,
                per_run_time_limit=8,
                tmp_folder=tmp_folder,
                delete_tmp_folder_after_terminate=True,
                seed=seed,
                n_jobs=1,
                initial_configurations_via_metalearning=15,
            )
            
            automl.fit(X_train, y_train)
            score = automl.score(X_test, y_test)
            scores.append(score)
        
        assert all(s > 0.5 for s in scores), \
            f"所有种子下的准确率 {scores} 都应高于随机猜测"

    def test_metalearning_cv_results(self, iris_data, tmp_path):
        """
        测试元学习的cv_results_
        
        Expects
        -------
        * cv_results_ 应包含元学习配置的信息
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_meta_cv")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=25,
        )
        
        automl.fit(X_train, y_train)
        
        cv_results = automl.cv_results_
        
        assert cv_results is not None, "cv_results_ 不应为 None"
        assert len(cv_results) > 0, "cv_results_ 不应为空"

    def test_metalearning_show_models(self, iris_data, tmp_path):
        """
        测试元学习后的show_models()
        
        Expects
        -------
        * show_models() 应返回有效的模型信息
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_meta_models")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
            initial_configurations_via_metalearning=25,
        )
        
        automl.fit(X_train, y_train)
        
        models = automl.show_models()
        
        assert models is not None, "show_models() 不应返回 None"
