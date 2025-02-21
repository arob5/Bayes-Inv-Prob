#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:39 2025

@author: vigneshsomjit
"""
from typing import Any

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
        
class ParamValue:
    """
    Encapsulates the actual value that a parameter can assume.
    """
    def __init__(self, 
                 param_info: ParamInfo,
                 value):
        
        self.param_info = param_info
        self._value = value 
        
        @property 
        def value(self) -> Any :
            return self._value 
        
        @value.setter 
        def setter(self, new_value: Any):
            self._value = new_value

class ParamGroup:
    """
    A container class that holds a collection of parameters (both their metadata and values).
    
    Each parameter is stored as a key-value pair, where: 
        - The key is the parameter's name
        - The value is a `ParamValue` object representing the parameter's current value
    """
    def __init__(self):
        """
        Initialize any empty parameter group (i.e. dictionary).
        """
        self._params = {}
        
    def add_parameter(self, param_info: ParamInfo, initial_value: Any):
        """
        Add a new parameter to the group.
        param_info: An instance of ParamInfo describing the parameter's metadata
        initial_value: The initial value assigned to that parameter 
        """
        if param_info.name in self._params:
            raise KeyError(f"Parameter '{param_info.name}' already exists in the group.")
        self._params[param_info.name] = ParamValue(value = initial_value, param_info = param_info)
    
    def remove_parameter(self, param_names: str | list | tuple):
        """
        Remove a parameter from the group by its name.
        """
        if isinstance(param_names, str):
            param_names = [param_names]  # Convert to string 
    
        if not isinstance(param_names, (list, tuple)):
            raise TypeError("param_names must be a string, list, or tuple.")
    
        for param_name in param_names:
            if param_name not in self._params:
                raise KeyError(f"No parameter named '{param_name}' in this group.")
            del self._params[param_name]
    
    def get_names(self) -> list[str]:
        """
        List the names of all parameters in the group.
        """
        return list(self._params.keys())

    
    
        