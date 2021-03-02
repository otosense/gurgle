from functools import wraps

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.
def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


class NoSuchItem:
    pass


no_such_item = NoSuchItem()


def cls_wrap(cls, obj):
    if isinstance(obj, type):

        @wraps(obj, updated=())
        class Wrap(cls):
            @wraps(obj.__init__)
            def __init__(self, *args, **kwargs):
                wrapped = obj(*args, **kwargs)
                super().__init__(wrapped)

        return Wrap
    else:
        return cls(obj)


class CallableModel:
    _call_method_str = 'transform'

    def __init__(self, model, *model_args, **model_kwargs):
        if isinstance(model, type):
            model = model(*model_args, **model_kwargs)
        self.model = model
        self.call_method = getattr(self.model, self._call_method_str)
        # self.fit = wraps(model.fit)

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        return getattr(self.model, attr)

    def __dir__(self):
        return list(set(dir(self.__class__)).union(self.model.__dir__()))  # to forward dir to delegated stream

    wrap = classmethod(cls_wrap)

    def fit(self, X, y=None, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)
        return self

    
    def preproc(self, x):
        return np.array([x])

    
    def postproc(self, x, model_output):
        return model_output[0]

    def __call__(self, x):
        x = self.preproc(x)
        model_output = self.call_method(x)
        post_output = self.postproc(x, model_output)
        return post_output


class CallableModelMixin:
    _call_method_str = 'transform'

    wrap = classmethod(cls_wrap)

    @property
    def call_method(self):
        return getattr(self, self._call_method_str)

    def fit(self, X, y=None, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)
        return self

    def preproc(self, x):
        return np.array([x])

    def postproc(self, x, model_output):
        return model_output[0]

    def __call__(self, x):
        x = self.preproc(x)
        model_output = self.call_method(x)
        post_output = self.postproc(x, model_output)
        return post_output


def enable_incremental_learning_on_call(call_method_str='transform',
                                        fv_is_a_single_number=False,
                                        warm_up=0,
                                        batch_size=0):
    """Makes a class creating instances that learn while they run (i.e. are applied to input data streams).

    A decorator meant to take any sklearn compatible Model that has a partial_fit and add
    (remove nothing -- just add) the ability to perform online learning (while being applied)
    simply by calling the instance with an fv .

    More precisely a call ``m(x)`` to an instance ``m`` will (basically):
    - call out = m.transform([x])  ("transform" by default, but could be any method)
    - partial_fit (every call, or buffer and partial fit in batches)
    - return out

    :param call_method_str:
    :return:

    >>> from sklearn.preprocessing import MinMaxScaler
    >>> @enable_incremental_learning_on_call()
    ... class OnlineMinMax(MinMaxScaler):
    ...     pass
    >>> m = OnlineMinMax()
    >>> m([0, 10, 20])
    >>> m([5, 10, 10])
    array([  5.,   0., -10.])
    >>> m([4, 25, 15])
    array([ 0.8, 15. ,  0.5])

    Sometimes the feature vector isn't a vector at all, but just a number, in which case this is more convenient:

    >>> @enable_incremental_learning_on_call(fv_is_a_single_number=True)
    ... class OnlineNumberMinMax(MinMaxScaler):
    ...     pass
    >>> m = OnlineNumberMinMax()
    >>> list(map(m, [0, 5, 4]))
    [None, 5.0, 0.8]

    >>> from sklearn.cluster import MiniBatchKMeans
    >>> @enable_incremental_learning_on_call(fv_is_a_single_number=True, batch_size=2)
    ... class Kms(MiniBatchKMeans):
    ...     pass
    >>> km = Kms(n_clusters=2, random_state=42)
    >>> list(map(km, [0, 5, 500, 10, 20, 1000, 2000]))  # remember that these are learned in batches (every 2)
    [None, None, 500.0, 10.0, 20.0, 1000.0, 1990.0]

    """

    def wrapper(cls):
        class Model(CallableModelMixin, cls):
            _call_method_str = call_method_str
            n_calls = 0
            n_warm_up_calls = warm_up
            not_fit_yet = None
            not_warm = None
            _batch_size = batch_size
            buffer = None

            def call_method(self, x):
                self.n_calls += 1
                if self.n_calls >= self.n_warm_up_calls:
                    try:
                        return super().call_method(x)
                    except NotFittedError:
                        return self.not_fit_yet
                else:
                    return self.not_warm

            @wraps(cls.partial_fit)
            def partial_fit(self, X, y=None, *args, **kwargs):
                if not self._batch_size:
                    super().partial_fit(X, y, *args, **kwargs)
                else:
                    self.batcherized_partial_fit(X, y, args, kwargs)
                return self

            def batcherized_partial_fit(self, X, y, args, kwargs):
                if self.buffer is None:
                    self.buffer = []
                # TODO: buffer len not intuitive (is num of items appended, not total number of X rows (points))
                self.buffer.append((X, y))
                if len(self.buffer) >= self._batch_size:
                    X, y = zip(*self.buffer)
                    X = np.vstack(X)
                    if all(yy is None for yy in y):
                        y = None
                    super().partial_fit(X, y, *args, **kwargs)
                    self.buffer = []  # re-initialize the buffer

            if not fv_is_a_single_number:
                def preproc(self, x):
                    return super().preproc(x)

                def postproc(self, x, model_output):
                    self.partial_fit(x)
                    if model_output is not self.not_fit_yet:
                        return super().postproc(x, model_output)
                    else:
                        return None
            else:
                def preproc(self, x):
                    return super().preproc([x])

                def postproc(self, x, model_output):
                    self.partial_fit(x)
                    if model_output is not self.not_fit_yet:
                        return super().postproc(x, model_output)[0]
                    else:
                        return self.not_fit_yet

        Model.__name__ = f'Online{cls.__name__}'

        return Model

    return wrapper


class TransparentModel(TransformerMixin, BaseEstimator):
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def partial_fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X):
        return X

    def predict_proba(self, X):
        return X

    def predict(self, X):
        return np.nan * np.zeros(X.shape[0])
