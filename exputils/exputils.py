"""
Handle experiment result naming schemes
"""

from typing import Mapping, Any, Container, Optional, ClassVar, Union, List
import itertools

import numpy as np

class BaseParameters:
    """
    Represent a parameter set used for running an experiment

    Parameters
    ----------
    parameters: Mapping[str, Any]
    """
    DATA_TYPES: ClassVar[Mapping[str, Any]] = {}
    STR_DELIMITER: ClassVar[str] = '--'

    def __init__(self, parameters: Mapping[str, Any]):
        if isinstance(parameters, dict):
            _data = self._init_from_dict(parameters)
        elif isinstance(parameters, BaseParameters):
            _data = self._init_from_dict(parameters.data)
        elif isinstance(parameters, str):
            _data = self._init_from_str(parameters)
        else:
            raise TypeError(f"Unknown type for `data` {type(parameters)}")

        # Check that the `DATA_TYPES` dict and input `data` dict
        # have 1-to-1 keys
        for key in _data:
            if key not in self.DATA_TYPES:
                raise ValueError(
                    f"{key} in input is not a valid parameter name"
                )

        for key in self.DATA_TYPES:
            if key not in _data:
                raise ValueError(
                    f"parameter name {key} was not found in input"
                )

        # Convert the parameter dictionary to the correct data types
        _data = {
            key: dtype(_data[key])
            for key, dtype in self.DATA_TYPES.items()
        }

        self._data = _data

    @classmethod
    def _init_from_dict(cls, data):
        """
        Return a parameter dictionary from an input dictionary
        """
        return data

    @classmethod
    def _init_from_str(cls, data_str):
        """
        Return a parameter dictionary from an input string
        """
        split_string = data_str.split(cls.STR_DELIMITER)
        parts_per_key = [
            len(dtype.DATA_TYPES) if issubclass(dtype, BaseParameters) else 1
            for dtype in cls.DATA_TYPES.values()
        ]
        nn = [x for x in itertools.accumulate(parts_per_key, initial=0)]
        str_parts = [
            cls.STR_DELIMITER.join(split_string[n0:n1])
            for n0, n1 in zip(nn[:-1], nn[1:])
        ]
        label_lengths = [len(key) for key in cls.DATA_TYPES]
        str_labels = [
            str_part[:n] for str_part, n in zip(str_parts, label_lengths)
        ]
        str_values = [
            str_part[n:]
            for str_part, n in zip(str_parts, label_lengths)
        ]

        data_str = {
            key:
                dtype._init_from_str(str_value) if issubclass(dtype, BaseParameters)
                else str_value
            for key, str_value, dtype in zip(str_labels, str_values, cls.DATA_TYPES.values())
        }
        return data_str

    @property
    def data(self):
        """
        Return the underlying dictionary of parameter keys and values
        """
        # Recursively get the dictionary representation of the parameters
        # and any sub-parameters
        return self._data

    @property
    def rdata(self):
        """
        Return the recursive underlying dictionary of parameter keys and values
        """
        # Recursively get the dictionary representation of the parameters
        # and any sub-parameters
        _data = {
            key:
                value.rdata if isinstance(value, BaseParameters)
                else value
            for key, value in self._data.items()
        }
        return _data

    ## Create derived instances
    def substitute(self, new_params: Mapping[str, any]) -> 'BaseParameters':
        """
        Return a new parameter set with different values

        Parameters
        ----------
        new_params :
        """
        new_data = self.data.copy()

        root_params = {
            key: value for key, value in new_params.items()
            if '/' not in key
        }
        for key, value in root_params.items():
            new_data[key] = value

        nest_params = {
            key: value for key, value in new_params.items()
            if '/' in key
        }
        nest_root_keys = {
            key.split('/')[0] for key in nest_params.keys()
        }
        nest_keys = {
            root_key: [] for root_key in nest_root_keys
        }
        for key in nest_params:
            split_key = key.split('/')
            root_key = split_key[0]
            sub_key = '/'.join(split_key[1:])
            nest_keys[root_key].append(sub_key)

        for root_key, sub_keys in nest_keys.items():
            _new_sub_params = {
                sub_key: nest_params[f'{root_key}/{sub_key}']
                for sub_key in sub_keys
            }
            new_data[root_key] = new_data[root_key].substitute(_new_sub_params)

        return self.__class__(new_data)

    ## String interface
    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def to_str(self, keys: Optional[Container[str]]=None):
        """
        Return a string representation of the parameters

        Parameters
        ----------
        keys : Container[str]
        """
        if keys is None:
            keys = self.DATA_TYPES.keys()
        return self.STR_DELIMITER.join([
            self._format_key_to_str(key) for key in self.DATA_TYPES
            if key in keys
        ])

    def _format_key_to_str(self, key: str) -> str:
        """
        Return a string representing the parameter indicated by `key`
        """
        dtype = self.DATA_TYPES[key]
        value = self[key]
        return f'{key}{self._format_value_to_str(value, dtype)}'

    def _format_value_to_str(
            self,
            value: Any,
            dtype: Optional[type]=None
        ) -> str:
        """
        Return a string representing a parameter value
        """
        if dtype is None:
            dtype = type(value)

        if issubclass(dtype, str):
            return value
        elif issubclass(dtype, float):
            return f'{value:.2e}'
        elif issubclass(dtype, bool):
            return f'{value:d}'
        elif issubclass(dtype, int):
            return f'{value:d}'
        elif issubclass(dtype, BaseParameters):
            return value.to_str()
        else:
            raise TypeError(f"Unknown `value` type {dtype}")

    ## Dictionary interface
    def keys(self):
        """
        Return dict keys
        """
        return self.data.keys()

    def values(self):
        """
        Return dict values
        """
        return self.data.values()

    def __contains__(self, key):
        """
        Return whether dict contains the given key
        """
        sub_keys = key.split('/')
        if len(sub_keys) > 1:
            if sub_keys[0] in self.data:
                return '/'.join(sub_keys[1:]) in self.data[sub_keys[0]]
            else:
                return False
        else:
            if key in self.data:
                return True
            else:
                return False

    def items(self):
        """
        Return a `dict.items` like iterator
        """
        return self.data.items()

    def __getitem__(self, key):
        split_keys = key.split('/')

        root_key = split_keys[0]
        if len(split_keys) == 1:
            return self._data[root_key]
        else:
            return self._data[root_key]['/'.join(split_keys[1:])]

    def __setitem__(self, key, value):
        raise NotImplementedError("Can't set parameter values")

    def __len__(self):
        return len(self.data)


def make_parameters(data_types: Mapping[str, type]):
    """
    Return a class representing a parameter set

    Parameters
    ----------
    data_types: Mapping[str, type]
        A specification of the parameter labels and data types
    """

    class Parameters(BaseParameters):
        DATA_TYPES = data_types

    return Parameters

ParamValues = Union[List[Any], Any]
def iter_parameters(
        substitute_params: Mapping[str, ParamValues],
        default_params: BaseParameters
    ):
    """
    Return an iterator over `BaseParameters` instances

    Parameters
    ----------
    substitute_params :
        A mapping from parameter labels to parameter value(s). Parameter
        instances are created for each parameter value given in the dictionary.
    defaults : ExpParam
        Default values for unspecified parameters
    """
    # Validate that all kwargs keys are valid parameters
    keys = list(substitute_params.keys())
    for key in keys:
        assert key in default_params

    # Make sure all parameter values are lists
    # (so that you can iterate over them)
    def _require_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return x
        else:
            return [x]

    _substitute_params = {
        key: _require_list(values)
        for key, values in substitute_params.items()
    }

    for vals in itertools.product(*list(_substitute_params.values())):
        yield default_params.substitute(
            {key: val for key, val in zip(keys, vals)}
        )


if __name__ == '__main__':
    ## Test case for a simple parameter set
    param_dtypes = {
        'name': str,
        'x': float,
        'y': float,
        'nx': int,
        'ny': int
    }

    Params = make_parameters(param_dtypes)
    try:
        p = Params({'x': 2, 'y': 3, 'use_a_diddlydoo': False})
    except ValueError as err:
        print(err)

    try:
        p = Params({'x': 2, 'y': 3})
    except ValueError as err:
        print(err)

    p = Params({'x':2, 'y': 3, 'nx':2, 'ny':5, 'name': 'gottfried'})
    print(p)

    ## Test case for a nested parameter set
    param_dtypes_child = {
        'childname': str,
        'x': float,
    }
    ChildParams = make_parameters(param_dtypes_child)

    param_dtypes_root = {
        'name': str,
        'x': float,
        'Child': ChildParams
    }
    RootParams = make_parameters(param_dtypes_root)

    try:
        p = RootParams({'name': 'gunther', 'x': 2, 'y': 3})
    except ValueError as err:
        print(err)

    try:
        p = RootParams({'name': 'gunther'})
    except ValueError as err:
        print(err)

    p = RootParams({'name': 'gunther', 'x': 2, 'Child': {'childname': 'guntherjr.', 'x': 2.1}})
    print(p)
    print(p.to_str())
    p2 = RootParams(p.to_str())
    print(p2)

    print(p2['name'])
    print(p2['x'])
    print(p2['Child/childname'])
    print(p2.substitute({'Child/childname': 'sally'}))
