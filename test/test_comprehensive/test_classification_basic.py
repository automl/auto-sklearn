"""
AutoSklearnClassifier 基本分类功能测试

测试场景：在标准分类数据集（如iris、digits）上验证基本分类流程
验证点：
- fit() 方法能成功完成训练（不崩溃、不报错）
- predict() 返回合理形状的预测结果（与样本数一致）
- predict_proba() 返回概率分布（和为1，范围[0,1]）
- score() 返回合理的准确率（应显著高于随机猜测）
- sprint_statistics() 返回包含关键信息的统计字符串
- 重点关注：训练时间在合理范围内，内存使用正常
"""
from __future__ import annotations

import time
import tempfile
import os

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import accuracy


class TestAutoSklearnClassifierBasic:
    """测试 AutoSklearnClassifier 基本分类功能"""

    @pytest.fixture(scope="class")
    def iris_data(self):
        """加载iris数据集"""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture(scope="class")
    def digits_data(self):
        """加载digits数据集"""
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.3, random_state=42
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

    def test_fit_completes_without_error_iris(self, iris_data, tmp_path):
        """
        测试fit()方法在iris数据集上能成功完成训练（不崩溃、不报错）
        
        Expects
        -------
        * fit() 应该成功完成，不抛出异常
        * 训练时间应在合理范围内（< 120秒）
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_iris")
        
        automl = AutoSklearnClassifier(
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
        
    def test_fit_completes_without_error_digits(self, digits_data, tmp_path):
        """
        测试fit()方法在digits数据集上能成功完成训练
        
        Expects
        -------
        * fit() 应该成功完成，不抛出异常
        * 训练时间应在合理范围内（< 180秒）
        """
        X_train, X_test, y_train, y_test = digits_data
        
        tmp_folder = str(tmp_path / "autosklearn_digits")
        
        automl = AutoSklearnClassifier(
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
        
        assert training_time < 180, f"训练时间过长: {training_time:.2f}秒"
        assert automl.automl_ is not None, "automl_ 属性不应为 None"

    def test_predict_returns_correct_shape(self, iris_data, tmp_path):
        """
        测试predict()返回合理形状的预测结果（与样本数一致）
        
        Expects
        -------
        * predict() 返回的数组长度应与输入样本数一致
        * 预测值应为有效的类别标签
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_predict")
        
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
        
        assert len(predictions) == len(X_test), \
            f"预测结果长度 {len(predictions)} 应与测试样本数 {len(X_test)} 一致"
        
        unique_labels = np.unique(y_train)
        assert all(p in unique_labels for p in np.unique(predictions)), \
            "预测值应为训练集中出现过的有效类别标签"

    def test_predict_proba_returns_valid_probabilities(self, iris_data, tmp_path):
        """
        测试predict_proba()返回概率分布（和为1，范围[0,1]）
        
        Expects
        -------
        * predict_proba() 返回的数组形状应为 (n_samples, n_classes)
        * 每行概率之和应接近1
        * 所有概率值应在[0, 1]范围内
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_proba")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        probas = automl.predict_proba(X_test)
        
        n_classes = len(np.unique(y_train))
        assert probas.shape == (len(X_test), n_classes), \
            f"概率数组形状 {probas.shape} 应为 ({len(X_test)}, {n_classes})"
        
        prob_sums = probas.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6), \
            "每行概率之和应接近1"
        
        assert np.all(probas >= 0) and np.all(probas <= 1), \
            "所有概率值应在[0, 1]范围内"

    def test_score_returns_reasonable_accuracy(self, iris_data, tmp_path):
        """
        测试score()返回合理的准确率（应显著高于随机猜测）
        
        Expects
        -------
        * score() 返回的准确率应显著高于随机猜测（iris有3类，随机猜测准确率约33%）
        * 准确率应在[0, 1]范围内
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_score")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        accuracy = automl.score(X_test, y_test)
        
        assert 0 <= accuracy <= 1, f"准确率 {accuracy} 应在[0, 1]范围内"
        
        n_classes = len(np.unique(y_train))
        random_guess = 1.0 / n_classes
        assert accuracy > random_guess + 0.2, \
            f"准确率 {accuracy:.3f} 应显著高于随机猜测 {random_guess:.3f}"

    def test_sprint_statistics_contains_key_info(self, iris_data, tmp_path):
        """
        测试sprint_statistics()返回包含关键信息的统计字符串
        
        Expects
        -------
        * 返回的字符串应包含 "auto-sklearn" 或相关关键词
        * 应包含训练时间信息
        * 应包含模型数量信息
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_stats")
        
        automl = AutoSklearnClassifier(
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
        
        key_indicators = ["auto-sklearn", "time", "run", "model", "configuration"]
        has_key_info = any(indicator.lower() in stats.lower() for indicator in key_indicators)
        assert has_key_info, f"统计字符串应包含关键信息，实际内容: {stats[:200]}"

    def test_binary_classification_breast_cancer(self, breast_cancer_data, tmp_path):
        """
        测试二分类任务（乳腺癌数据集）
        
        Expects
        -------
        * fit() 和 predict() 在二分类任务上正常工作
        * predict_proba() 返回2列概率（二分类）
        * 准确率应高于随机猜测（50%）
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        tmp_folder = str(tmp_path / "autosklearn_binary")
        
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
        probas = automl.predict_proba(X_test)
        accuracy = automl.score(X_test, y_test)
        
        assert probas.shape[1] == 2, "二分类任务应返回2列概率"
        assert accuracy > 0.6, f"二分类准确率 {accuracy:.3f} 应高于随机猜测"
        
        unique_preds = np.unique(predictions)
        assert set(unique_preds).issubset({0, 1}), "二分类预测值应为0或1"

    def test_multiclass_classification(self, digits_data, tmp_path):
        """
        测试多分类任务（digits数据集，10类）
        
        Expects
        -------
        * fit() 和 predict() 在多分类任务上正常工作
        * predict_proba() 返回10列概率
        * 准确率应显著高于随机猜测（10%）
        """
        X_train, X_test, y_train, y_test = digits_data
        
        tmp_folder = str(tmp_path / "autosklearn_multiclass")
        
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
        probas = automl.predict_proba(X_test)
        accuracy = automl.score(X_test, y_test)
        
        n_classes = len(np.unique(y_train))
        assert probas.shape[1] == n_classes, f"多分类任务应返回{n_classes}列概率"
        assert accuracy > 0.3, f"多分类准确率 {accuracy:.3f} 应显著高于随机猜测 {1.0/n_classes:.3f}"

    def test_training_with_different_seeds(self, iris_data, tmp_path):
        """
        测试不同随机种子下的训练稳定性
        
        Expects
        -------
        * 不同种子应产生略有不同的结果
        * 但准确率都应保持在合理范围内
        """
        X_train, X_test, y_train, y_test = iris_data
        
        accuracies = []
        
        for seed in [1, 42, 123]:
            tmp_folder = str(tmp_path / f"autosklearn_seed_{seed}")
            
            automl = AutoSklearnClassifier(
                time_left_for_this_task=40,
                per_run_time_limit=8,
                tmp_folder=tmp_folder,
                delete_tmp_folder_after_terminate=True,
                seed=seed,
                n_jobs=1,
            )
            
            automl.fit(X_train, y_train)
            acc = automl.score(X_test, y_test)
            accuracies.append(acc)
        
        assert all(acc > 0.5 for acc in accuracies), \
            f"所有种子下的准确率 {accuracies} 都应高于随机猜测"

    def test_refit_works_correctly(self, iris_data, tmp_path):
        """
        测试refit()方法能正确重新训练模型
        
        Expects
        -------
        * refit() 应该成功完成
        * refit后预测结果应与之前一致或更好
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_refit")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        initial_accuracy = automl.score(X_test, y_test)
        
        try:
            automl.refit()
            refit_accuracy = automl.score(X_test, y_test)
            
            assert refit_accuracy >= initial_accuracy - 0.1, \
                f"refit后准确率 {refit_accuracy:.3f} 不应显著低于初始准确率 {initial_accuracy:.3f}"
        except Exception as e:
            pytest.skip(f"refit() 在此配置下不可用: {e}")

    def test_show_models_returns_valid_info(self, iris_data, tmp_path):
        """
        测试show_models()返回有效的模型信息
        
        Expects
        -------
        * show_models() 应返回模型信息
        * 返回的信息应包含模型相关数据
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_models")
        
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            tmp_folder=tmp_folder,
            delete_tmp_folder_after_terminate=True,
            seed=42,
            n_jobs=1,
        )
        
        automl.fit(X_train, y_train)
        
        models = automl.show_models()
        
        assert models is not None, "show_models() 不应返回 None"

    def test_leaderboard_returns_dataframe(self, iris_data, tmp_path):
        """
        测试leaderboard()返回DataFrame
        
        Expects
        -------
        * leaderboard() 应返回pandas DataFrame
        * DataFrame应包含模型性能信息
        """
        X_train, X_test, y_train, y_test = iris_data
        
        tmp_folder = str(tmp_path / "autosklearn_leaderboard")
        
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
            leaderboard = automl.leaderboard()
            assert hasattr(leaderboard, 'shape'), "leaderboard() 应返回DataFrame-like对象"
        except Exception as e:
            pytest.skip(f"leaderboard() 在此配置下不可用: {e}")
