# auto-sklearn

auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

Find the documentation [here](https://github.com/automl/auto-sklearn/blob/master/source/index.rst)


# Схема работы 


- ...
- Запускается smac, который выполняет несколько раз runsolver. Runsolver в свою очередь запускает скрипт SMAC_cli_holdout.py
    - основным результатом этого шага является создание файлов-моделей
    - еще predictions_ensemble создается
- Запускается runsolver, который запускает несколько раз ensemble_selection_script.py
    - основной результата шага: создание ensemble_indices_1