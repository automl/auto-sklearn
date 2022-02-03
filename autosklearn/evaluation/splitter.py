import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _approximate_mode, check_random_state, indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_array, column_or_1d


class CustomStratifiedShuffleSplit(StratifiedShuffleSplit):
    """Splitter that deals with classes with too few samples"""

    def _iter_indices(self, X, y, groups=None):  # type: ignore
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)
            train = []
            test = []

            # NOTE: Adapting for unique instances
            #
            #   Each list n_i, t_i represent the list of class in the
            #   training_set and test_set resepectively.
            #
            #   n_i = [100, 100, 0, 3]  # 100 of class '0', 0 of class '2'
            #   t_i = [300, 300, 1, 3]  # 300 of class '0', 1 of class '2'
            #
            #  To support unique labels such as class '2', which only has one sample
            #  between both n_i and t_i, we need to make sure that n_i has at least
            #  one sample of all classes. There is also the extra check to ensure
            #  that the sizes stay the same.
            #
            #   n_i = [ 99, 100, 1, 3]  # 100 of class '0', 0 of class '2'
            #            |       ^
            #            v       |
            #   t_i = [301, 300, 0, 3]  # 300 of class '0', 1 of class '2'
            #
            for i, class_count in enumerate(n_i):
                if class_count == 0:
                    t_i[i] -= 1
                    n_i[i] += 1

                    j = np.argmax(n_i)
                    if n_i[j] == 1:
                        warnings.warn(
                            "Can't respect size requirements for split.",
                            " The training set must contain all of the unique"
                            " labels that exist in the dataset.",
                        )
                    else:
                        n_i[j] -= 1
                        t_i[j] += 1

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test


class CustomStratifiedKFold(StratifiedKFold):
    """Stratified K-Folds cross-validator that ensures that there is always at least
    1 sample per class in the training set.
    """

    def _make_test_folds(self, X, y=None):  # type: ignore
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def split(self, X, y=None, groups=None):  # type: ignore

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )

        for train, test in super().split(X, y, groups):
            all_classes = np.unique(y)
            train_classes = np.unique(y[train])
            train = list(train)
            test = list(test)
            missing_classes = set(all_classes) - set(train_classes)
            if len(missing_classes) > 0:
                # print(missing_classes)
                for diff in missing_classes:
                    # print(len(train), len(test))
                    to_move = np.where(y[test] == diff)[0][0]
                    # print(y[test][to_move])
                    train = train + [test[to_move]]
                    del test[to_move]
                    # print(len(train), len(test))
            train = np.array(train, dtype=int)
            test = np.array(test, dtype=int)

            yield train, test
