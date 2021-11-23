# auto-sklearn

**auto-sklearn** is an automated machine learning toolkit and a drop-in replacement for a [scikit-learn](https://scikit-learn.org) estimator.

Find the documentation **[here](https://automl.github.io/auto-sklearn/)**. Quick links:
  * [Installation Guide](https://automl.github.io/auto-sklearn/master/installation.html)
  * [Releases](https://automl.github.io/auto-sklearn/master/releases.html)
  * [Manual](https://automl.github.io/auto-sklearn/master/manual.html)
  * [Examples](https://automl.github.io/auto-sklearn/master/examples/index.html)
  * [API](https://automl.github.io/auto-sklearn/master/api.html)

## auto-sklearn in one image

![image](doc/images/askl_pipeline.png)

## auto-sklearn in four lines of code

```python
import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
```

## auto-sklearn acceleration with sklearnex

You can accelerate auto-sklearn with [Intel(R) Extension for Scikit-Learn (sklearnex)](https://github.com/intel/scikit-learn-intelex). The acceleration is achieved through **patching**: replacing scikit-learn algorithms with their optimized versions provided by the extension.

Install sklearnex with pip or conda:
```bash
pip install scikit-learn-intelex
conda install scikit-learn-intelex
conda install -c conda-forge scikit-learn-intelex
```

To accelerate auto-sklearn, insert the following two lines of patching code before auto-sklearn and sklearn imports:

```python
from sklearnex import patch_sklearn
patch_sklearn()

import autosklearn.classification
```

To return to the original scikit-learn implementation, unpatch scikit-learn and reimport auto-sklearn and sklearn:

```python
from sklearnex import unpatch_sklearn
unpatch_sklearn()

import autosklearn.classification
```

## Relevant publications

If you use auto-sklearn in scientific publications, we would appreciate citations.

**Efficient and Robust Automated Machine Learning**  
*Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter*  
Advances in Neural Information Processing Systems 28 (2015)  

[Link](https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf) to publication.
```
@inproceedings{feurer-neurips15a,
    title     = {Efficient and Robust Automated Machine Learning},
    author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina  Springenberg, Jost and Blum, Manuel and Hutter, Frank},
    booktitle = {Advances in Neural Information Processing Systems 28 (2015)},
    pages     = {2962--2970},
    year      = {2015}
}
```

----------------------------------------

**Auto-Sklearn 2.0: The Next Generation**  
*Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter**  
arXiv:2007.04074 [cs.LG], 2020

[Link](https://arxiv.org/abs/2007.04074) to publication.
```
@article{feurer-arxiv20a,
    title     = {Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning},
    author    = {Feurer, Matthias and Eggensperger, Katharina and Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
    booktitle = {arXiv:2007.04074 [cs.LG]},
    year      = {2020}
}
```

----------------------------------------

Also, have a look at the blog on [automl.org](https://automl.org) where we regularly release blogposts.
