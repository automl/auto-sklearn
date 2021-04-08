# auto-sklearn

auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

Find the documentation [here](http://automl.github.io/auto-sklearn/)

## Automated Machine Learning in four lines of code

```python
import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
```

## Relevant publications

Efficient and Robust Automated Machine Learning  
Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter  
Advances in Neural Information Processing Systems 28 (2015)  
http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf

Auto-Sklearn 2.0: The Next Generation  
Authors: Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter  
arXiv:2007.04074 [cs.LG], 2020
https://arxiv.org/abs/2007.04074
