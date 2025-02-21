#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:39 2025

@author: vigneshsomjit
"""
from typing import Any
import numpy as np

class ParamInfo:
    """
    Encapsulates metadata for a parameter.
    """
    def __init__(self,
                 name: str,
                 param_type: str,
                 constraints: dict,
                 shape: int | tuple[int, ...]):
        """
        name: Name of the parameter.
        param_type: Type of the parameter (e.g. 'float', 'int', 'categorical', etc.).
        constraints: Dictionary of constraints (e.g. {'min': 0.0, 'max': 10.0}).
        shape: Expected shape of the parameter.
        """
        self.name = name
        self.param_type = param_type
        self.constraints = constraints
        self.shape = shape


class ParamGroup:
    """
    A container class that holds a collection of parameters (both their metadata
    and values).

    Each parameter is stored as a key-value pair, where:
        - The key is the parameter's name
        - The value is a `ParamInfo` object storing the parameter metadata.
    """

    def __init__(self, param_info: Dict):
        """

        Parameters
        ----------
        param_info : `Dict`
            Dictionary of information for all parameters in the group. Keys are
            parameter names and values are themselves dictionaries. The inner
            dictionaries must have keys "type", "constraint", "size".
        """
        self._param_info = param_info

    def get_param_names(self, include_arr_names=False) -> list[str]:
        """ Return list of parameter names in alphabetical order.

        Parameters
        ----------
        include_arr_names : `bool`
            If False, then list consists of the set of keys in `self.param_info`.
            If `include_arr_names` is True then the individual elements of
            array-valued parameters are included.
        """
        if include_arr_names:
            raise NotImplementedError()
        else:
            return sorted(self._param_info.keys())

    def add_param(self, param_name: str, param_info: Dict):
        """ Add a single new parameter to the group.

        Parameters
        ----------
        param_name : `str`
            The new parameter name.
        param_info : `Dict`
            The parameter information dictionary for a new parameter.
        """
        if param_name in self._param_info.keys():
            raise KeyError(f"Parameter {param_name} already exists in the group.")
        self.param_info.update(param_name=param_info)

    def remove_param(self, param_names: str | list | tuple):
        """ Remove one or more parameters from the group by name.

        Parameters
        ----------
        param_names : `str`, `list`, or `tuple`
            The parameter name(s) to remove.
        """
        if isinstance(param_names, str):
            param_names = [param_names]  # Convert to list

        if not isinstance(param_names, (list, tuple)):
            raise TypeError("param_names must be a string, list, or tuple.")

        for param_name in param_names:
            self._param_info.pop(param_name)


class ParamValue:
    """
    Encapsulates the actual value that a parameter can assume.
    """
    def __init__(self, param: ParamGroup, val: Dict = None): #
        """

        Parameters
        ----------
        param : `ParamGroup`
            Instance of ParamGroup class, defining the parameter structure.
        val : `Dict`
            Dictionary storing the values. Keys are parameter names and values
            are the parameter values.
        """
        self.param = param
        self.value = val

    @property
    def value(self) -> Any :
        return self._value

    @value.setter
    def value(self, val: Any):
        # TODO: add checks here for param type/size/constraints. Should
        # probably allow `val` to be None as well.
        self._value = val

    @value.deleter
    def value(self):
        del self._value

    def to_array():
        raise NotImplementedError()
