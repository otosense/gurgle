import numpy as np


def row_euclidean_distance(A, B):
    """Euclidean distance between aligned rows of A. An array of length len(A) (==len(B)).

    >>> import numpy as np
    >>> A = np.arange(5 * 16).reshape((5, 16))
    >>> B = 1 + A

    >>> assert all(row_euclidean_distance(A, A) == np.zeros(5))
    >>> assert all(row_euclidean_distance(A, B) == np.array([4., 4., 4., 4., 4.]))

    Note: Not to be confused with the matrix of distances of all pairs of rows. Here, equivalent to the latter diagnonal (see below).

    ```
    from  sklearn.metrics.pairwise import euclidean_distances
    A = np.random.rand(5, 7)
    B = np.random.rand(5, 7)
    assert all(np.diag(euclidean_distances(A, B)) == row_euclidean_distance(A, B))
    ```

    """
    return np.sqrt(((A - B) ** 2).sum(axis=1))


from functools import wraps


# TODO: Add possibility to specify inclusion/exclusion lists
# TODO: If func arg is annotated with "str", don't try to replace it
# TODO: Raise specific error so outside use can catch and handle in the knowledge that replacedments were done.
def resolve_str_specification(obj_store):
    def _resolve_str_specification(func):
        @wraps(func)
        def _func(*args, **kwargs):
            def _args():
                for a in args:
                    if isinstance(a, str) and a in obj_store:
                        yield obj_store[a]
                    else:
                        yield a

            def _kwargs():
                for k, v in kwargs.items():
                    if isinstance(v, str) and v in obj_store:
                        yield k, obj_store[v]
                    else:
                        yield k, v

            return func(*_args(), **dict(_kwargs()))

        return _func

    return _resolve_str_specification
