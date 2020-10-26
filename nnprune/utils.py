from typing import TypeVar, Union, Tuple


_T = TypeVar('T')


def cast_tuple_to_scalar(v: Union[_T, Tuple[_T, ...]]):
    if isinstance(v, tuple):
        for i in range(1, len(v)):
            assert v[i] == v[0]
        return v[0]
    else:
        return v
