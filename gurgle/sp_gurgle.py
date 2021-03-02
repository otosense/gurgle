from functools import wraps, cached_property
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import TransformerMixin, BaseEstimator

from gurgle.base import CallableModel, TransparentModel


def instantiate_if_type(obj):
    if isinstance(obj, type):
        return obj()
    else:
        return obj


def orth(A):
    """
    Orthonormalize the matrix A.

    Note: Code is more complicated than the purest mathematical need because of numerical problem mitigation.

    :param A: a n-by-k array
    :return: a n-by-k array whose row spans the same space as the rows of A and with the normal rows

    >>> M = np.array([[1,2,3], [3,4,5]])
    >>> O = orth(M)
    >>> # checking the product of O and its transpose is the identity
    >>> np.allclose(np.dot(O, O.T), np.eye(2))
    True

    """
    u, s, vh = np.linalg.svd(A.T, full_matrices=False)
    N, M = A.shape
    eps = np.finfo(float).eps
    tol = max(np.max(M), np.max(N)) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q.T


def space_residue(self, X):
    return X - np.dot(X, self.orth_projection)


def sub_space_residue(self, X):
    return np.dot(self.space_residue(X), self.orth_scaling)


class ScaledPCA(PCA):
    scaler_cls = StandardScaler
    scaler_ = None

    @wraps(PCA.fit)
    def fit(self, X, y=None, *args, **kwarg):
        self.scaler_ = self.scaler_cls().fit(X)
        super().fit(self.scaler_.transform(X), y=None, *args, **kwarg)
        return self

    @wraps(PCA.transform)
    def transform(self, X):
        return super().transform(self.scaler_.transform(X))

    @wraps(PCA.inverse_transform)
    def inverse_transform(self, X):
        return self.scaler_.inverse_transform(super().inverse_transform(X))


class ProjectorResiduesOnCall(CallableModel):
    def __init__(self, model=ScaledPCA, call_method='transform', score_scaler=TransparentModel):
        super().__init__(model, call_method)
        self.score_scaler = instantiate_if_type(score_scaler)
        self._x_buffer = list()

    # @cached_property
    # def projection(self):
    #     return np.dot(self.scalings_, self.scalings_.T)
    #
    # @cached_property
    # def orth_projection(self):
    #     return orth(self)

    def is_fit(self):
        return hasattr(self.model, 'components_') and self.model.components_ is not None

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        # TODO: Make this independent of self.model.components_ (should only use X and self.model.transform(X))
        #   so that ScaledPCA can work as well.
        if hasattr(self.model, 'scaler_'):  # TODO: Get rid of this hack by learning some damn linear algebra!
            return np.sum(cdist(self.model.scaler_.transform(X), self.model.components_), axis=1).ravel()
        else:
            return np.sum(cdist(X, self.model.components_), axis=1).ravel()

    def fit(self, X, y=None, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)
        scores = self.decision_function(X)
        self.score_scaler.fit(scores.reshape((-1, 1)))
        return self

    def postproc(self, X, model_output):
        fv, scaled_score = None, None
        if self.is_fit():
            scores = self.decision_function(X)
            scaled_scores = self.score_scaler.transform(scores.reshape((-1, 1)))
            fv, scaled_score = X[0], scaled_scores[0]
        return fv, scaled_score


class IncrementalProjectorResiduesOnCall(ProjectorResiduesOnCall):
    def __init__(self, model=IncrementalPCA, call_method='transform', score_scaler=StandardScaler, learn_batch_size=1):
        super().__init__(model, call_method)
        self.score_scaler = score_scaler
        self.learn_batch_size = learn_batch_size
        self._x_buffer = list()

    def preproc(self, x):
        self._x_buffer.append(x)
        return np.array([x])

    def postproc(self, X, model_output):
        fv, scaled_score = super().postproc(X, model_output)
        if len(self._x_buffer) >= self.learn_batch_size:
            self.model.partial_fit(np.array(self._x_buffer))  # learn a batch
            self._x_buffer = list()  # reset buffer
        return fv, scaled_score


def partial_fittable_instance(obj, name=None, **kwargs):
    """Asserts that obj has a partial_fit, and instantiates it with **kwargs if obj is given as a type"""
    if name is None:
        if isinstance(obj, type):
            name = obj.__name__
        else:
            name = type(obj).__name__

    assert hasattr(obj, 'partial_fit'), f"{name} doesn't have a partial_fit: {obj} "
    if isinstance(obj, type):
        obj = obj(**kwargs)
    return obj


class ScaledIncrementalPCA(IncrementalPCA, ScaledPCA):
    scaler_cls = StandardScaler
    scaler_ = None

    def is_fit(self):
        return hasattr(self, 'components_') and self.components_ is not None

    @wraps(IncrementalPCA.partial_fit)
    def partial_fit(self, X, y=None, check_input=True):
        if self.scaler_ is None:
            self.scaler_ = self.scaler_cls()

        if not self.is_fit():
            # then this is the first training data piece we see, so we need to make sure we have enough to start fitting
            nrows, fv_size = X.shape
            if self.n_components is None:
                self.n_components = fv_size
            X = complete_with_fake_data_for_warmup(minimum_n_rows_to_fit=self.n_components, X=X)
        self.scaler_.partial_fit(X)
        scaled_X = self.scaler_.transform(X)
        super().partial_fit(scaled_X, y, check_input=check_input)
        return self


class ResidueGurgle(BaseEstimator):
    learn_batch_size = None

    def __init__(self, projector=ScaledIncrementalPCA, score_scaler=StandardScaler):
        # assert n_components is not None, "You need to specify an actual number of components. No None allowed here."
        self.projector = partial_fittable_instance(projector, 'projector')
        self.score_scaler = partial_fittable_instance(score_scaler, 'score_scaler')

        self._x_buffer = list()
        self.learn_batch_size = self.learn_batch_size or (self.n_components + 1)

    @property
    def n_components(self):
        return self.projector.n_components

    def is_fit(self):
        return hasattr(self.projector, 'components_') and self.projector.components_ is not None

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        """
        # check_is_fitted(self, ['components_'])
        # X = check_array(X)
        # return np.sum(cdist(self.projector, self.projector.components_), axis=1).ravel()
        # TODO: Make this independent of self.model.components_ (should only use X and self.model.transform(X))
        #   so that ScaledPCA can work as well.
        if hasattr(self.projector, 'scaler_'):  # TODO: Get rid of this hack by learning some damn linear algebra!
            return np.sum(cdist(self.projector.scaler_.transform(X), self.projector.components_), axis=1).ravel()
        else:
            return np.sum(cdist(X, self.projector.components_), axis=1).ravel()

    def partial_fit(self, X, y=None):
        if not self.is_fit():
            # then this is the first training data piece we see, so we need to make sure we have enough to start fitting
            nrows, fv_size = X.shape
            if self.n_components is None:
                self.projector.n_components = fv_size
            X = complete_with_fake_data_for_warmup(minimum_n_rows_to_fit=self.n_components, X=X)
        self.projector.partial_fit(X, y)
        scores = self.decision_function(X)
        self.score_scaler.partial_fit(scores.reshape((-1, 1)))
        return self

    def __call__(self, x):
        fv, scaled_score = None, None
        X = np.array([x])
        self._x_buffer.extend(X.tolist())
        if self.is_fit():
            fvs = self.projector.transform(X)
            scores = self.decision_function(X)
            scaled_scores = self.score_scaler.transform(scores.reshape((-1, 1)))
            fv, scaled_score = fvs[0], scaled_scores[0]
        if len(self._x_buffer) >= (self.learn_batch_size or len(x)):
            self.partial_fit(np.array(self._x_buffer))  # learn a batch
            self._x_buffer = list()  # reset buffer
        return fv, scaled_score

    #
    # def __call__(self, x):
    #     x = self.preproc(x)
    #     model_output = self.call_method(x)
    #     post_output = self.postproc(x, model_output)
    #     return post_output

def complete_with_fake_data_for_warmup(minimum_n_rows_to_fit, X=None, fv_size=None):
    """Makes fake data to warmup a partial fit process.
    If no X is given, will return a random minimum_n_rows_to_fit x fv_size matrix (with values between 0 and 1)
    If X is given, will repeat the rows in a cycle until the minimum_n_rows_to_fit is reached

    >>> X = complete_with_fake_data_for_warmup(3, fv_size=2);
    >>> X.shape
    (3, 2)
    >>> import numpy as np
    >>> complete_with_fake_data_for_warmup(5, X=np.array([[1,2,3], [4,5,6]]))
    array([[1, 2, 3],
           [4, 5, 6],
           [1, 2, 3],
           [4, 5, 6],
           [1, 2, 3]])
    """
    if X is None:
        assert fv_size is not None, "You need to have some data, or specify an fv_size"
        return np.random.rand(minimum_n_rows_to_fit, fv_size)
    else:
        nrows, fv_size = X.shape
    missing_n_rows = max(0, minimum_n_rows_to_fit - nrows)
    if missing_n_rows > 0:
        return np.array(X.tolist() * int(1 + np.ceil(missing_n_rows / nrows)))[:minimum_n_rows_to_fit]
    else:
        return X


############## Not used at the moment ##################################################################################


class GurgleWrap(BaseEstimator):
    learn_batch_size = None

    def __init__(self, model, score_scaler=StandardScaler, learn_batch_size=1):
        # assert n_components is not None, "You need to specify an actual number of components. No None allowed here."
        self.model = partial_fittable_instance(model, 'model')
        self.score_scaler = partial_fittable_instance(score_scaler, 'score_scaler')

        self._x_buffer = list()
        self.learn_batch_size = learn_batch_size

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        """
        # check_is_fitted(self, ['components_'])
        X = check_array(X)
        return np.sum(cdist(X, self.projector.components_), axis=1).ravel()

    def partial_fit(self, X, y=None):
        if not self.is_fit():
            # then this is the first training data piece we see, so we need to make sure we have enough to start fitting
            nrows, fv_size = X.shape
            if self.n_components is None:
                self.projector.n_components = fv_size
            X = complete_with_fake_data_for_warmup(minimum_n_rows_to_fit=self.n_components, X=X)
        self.projector.partial_fit(X, y)
        scores = self.decision_function(X)
        self.score_scaler.partial_fit(scores.reshape((-1, 1)))
        return self

    def __call__(self, x):
        fv, scaled_score = None, None
        X = np.array([x])
        self._x_buffer.extend(X.tolist())
        if self.is_fit():
            fvs = self.projector.transform(X)
            scores = self.decision_function(X)
            scaled_scores = self.score_scaler.transform(scores.reshape((-1, 1)))
            fv, scaled_score = fvs[0], scaled_scores[0]
        if len(self._x_buffer) >= (self.learn_batch_size or len(x)):
            self.partial_fit(np.array(self._x_buffer))  # learn a batch
            self._x_buffer = list()  # reset buffer
        return fv, scaled_score


class PartialFitPipeline:
    def __init__(self, *steps):
        self.steps = steps
        self.last_step = steps[-1]

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        return getattr(self.last_step, attr)

    def transform_until_last_step(self, X):
        for step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def partial_fit_transform(self, X, y=None, **kwargs):
        for step in self.steps:
            X = step.partial_fit(X, y, **kwargs).transform(X)
        return X

    def partial_fit(self, X, y=None, **kwargs):
        self.partial_fit_transform(X, y, **kwargs)
        return self

    def fit(self, X, y=None, **kwargs):
        for step in self.steps[:-1]:
            X = step.fit(X, y, **kwargs).transform(X)
        self.last_step.fit(X)
        return self

    def transform(self, X):
        return self.last_step.transform(self.transform_until_last_step(X))

    def predict_proba(self, X):
        return self.last_step.predict_proba(self.transform_until_last_step(X))

    def predict(self, X):
        return self.last_step.predict(self.transform_until_last_step(X))


class ProjectionDecisionFuncMixin:
    components_ = None  # lint appeaser

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['components_'])
        X = check_array(X)
        return np.sum(cdist(X, self.components_), axis=1).ravel()

# class ScaledPCA(PCA):
#     scaler_cls = StandardScaler
#     scaler_ = None
#
#     @wraps(PCA.fit)
#     def fit(self, X, y=None, *args, **kwarg):
#         self.scaler_ = self.scaler_cls().fit(X)
#         super().fit(self.scaler_.transform(X), y=None, *args, **kwarg)
#         return self
#
#     @wraps(PCA.transform)
#     def transform(self, X):
#         return super().transform(self.scaler_.transform(X))
#
#     @wraps(PCA.inverse_transform)
#     def inverse_transform(self, X):
#         return self.scaler_.inverse_transform(super().inverse_transform(X))


# class PcaGurgle(ScaledIncrementalPCA, ProjectionDecisionFuncMixin):
#     learn_batch_size = None
#
#     @wraps(ScaledIncrementalPCA.__init__)
#     def __init__(self, n_components, score_scaler=StandardScaler, *args, **kwargs):
#         # assert n_components is not None, "You need to specify an actual number of components. No None allowed here."
#         super().__init__(n_components, *args, **kwargs)
#         self.score_scaler = partial_fittable_instance(score_scaler, 'score_scaler')
#         self._x_buffer = list()
#         self.learn_batch_size = self.learn_batch_size or self.n_components
#
#     def is_fit(self):
#         return hasattr(self, 'components_') and self.components_ is not None
#
#     def partial_fit(self, X, y=None, check_input=True):
#         if not self.is_fit():
#             # then this is the first training data piece we see, so we need to make sure we have enough to start fitting
#             nrows, fv_size = X.shape
#             if self.n_components is None:
#                 self.n_components = fv_size
#             X = complete_with_fake_data_for_warmup(minimum_n_rows_to_fit=self.n_components, X=X)
#         super().partial_fit(X, y, check_input)
#         scores = self.decision_function(X)
#         self.score_scaler.partial_fit(scores.reshape((-1, 1)))
#         return self
#
#     def __call__(self, x):
#         fv, scaled_score = None, None
#         X = np.array([x])
#         self._x_buffer.extend(X.tolist())
#         if self.is_fit():
#             fvs = self.transform(X)
#             scores = self.decision_function(X)
#             scaled_scores = self.score_scaler.transform(scores.reshape((-1, 1)))
#             fv, scaled_score = fvs[0], scaled_scores[0]
#         if len(self._x_buffer) >= (self.learn_batch_size or len(x)):
#             self.partial_fit(np.array(self._x_buffer))  # learn a batch
#             self._x_buffer = list()  # reset buffer
#         return fv, scaled_score

# if self._x_buffer
# X = np.array([x])  # .resize((1, -1))
# if self.is_fit():
#     fvs = self.transform(X)
# else:
#     fvs = self.partial_fit(X).transform(X)
# scores = self.decision_function(X)
# scaled_scores = self.score_scaler.transform(scores.reshape((-1, 1)))
# self.partial_fit(X)
# return fvs[0], scaled_scores[0]
