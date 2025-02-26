#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:39 2025

@author: vigneshsomjit
"""
import numpy as np
import copy

'''
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
'''

class ParamGroup:
    """ 
    Holds a collection of parameters (both their metadata and values).
    """

    def __init__(self, param_info):
        """

        Parameters
        ----------
        param_info : `Dict`
            Nested dictionary storing information for all parameters in the group. 
            Outer keys are parameter names and values are dictionaries containing 
            paramater metadata. The inner keys must have "type", "constraint", "size".
        """
        self.param_info = copy.deepcopy(param_info) # Prevent unintended changes

    def get_param_names(self, include_arr_names=False):
        """ Return list of parameter names in alphabetical order.

        Parameters
        ----------
        include_arr_names : `bool`
            If False, then list consists of the set of keys in `self.param_info`.
            If True, then the individual elements of array-valued parameters are included.
        """
        if include_arr_names:
            raise NotImplementedError()
        else:
            return sorted(list(self.param_info.keys()))

    def add_param(self, param_name, param_metadata):
        """ Add a single new parameter to the group.

        Parameters
        ----------
        param_name : `str`
            The new parameter name.
        param_metadata : `Dict`
            The parameter information dictionary for a new parameter.
        """
        if param_name in self.param_info:
            raise KeyError(f"Parameter '{param_name}' already exists in the group.")
        self.param_info[param_name] = param_metadata
        
    def remove_param(self, param_names):
        """ Remove one or more parameters from the group by name.

        Parameters
        ----------
        param_names : `str`, `list`, or `tuple`
            The parameter name(s) to remove.
        """
        if isinstance(param_names, str):
            param_names = [param_names]  

        if not isinstance(param_names, (list, tuple)):
            raise TypeError("param_names must be a string, list, or tuple.")

        for param_name in param_names:
            if param_name not in self.param_info:
                raise KeyError(f"Parameter '{param_name}' does not exist in the group.")
            self.param_info.pop(param_name)



class ParamValue:
    """
    Encapsulates the actual value that a parameter can assume.
    """
    def __init__(self, param, init_values=None):
        """

        Parameters
        ----------
        param : `ParamGroup`
            Instance of ParamGroup class, defining the parameter structure.
            This is helpful for validation purposes. 
        init_values : `Dict`
            Dictionary storing the initial values. Keys are parameter names and 
            values are the parameter values.
        """
        self.param = param
        self._value = {} # Initialize internal storage of parameter values 
        
        if init_values is not None:
            self.value = init_values # Call the setter to validate init_values
    
    @staticmethod 
    def _validate_keys(expected_names, candidate_names):
        """
        Validates that the number of parameter values given equal the number 
        of parameters in the parameter group. 
        """
        # Check for extra parameters 
        extra_params = candidate_names - expected_names
        if extra_params:
            raise KeyError(f"Unexpected parameters in values: {extra_params}")
        # Check for missing parameters 
        missing_params = expected_names - candidate_names 
        if missing_params:
            raise KeyError(f"Missing values for parameters: {missing_params}")
    
    @staticmethod
    def _validate_type(name, value, expected_type):
        """
        Validates the type of the parameter value. Expected types include:
            - Float 
            - Integer
        """
        # Float type
        if expected_type == "float" and not isinstance(value, (float, np.ndarray)):
            raise TypeError(f"Parameter '{name}' should be of type {expected_type}, got {type(value)}")
        # Integer type 
        if expected_type == "int" and not isinstance(value, int):
            raise TypeError(f"Parameter '{name}' should be of type {expected_type}, got {type(value)}")
    
    @staticmethod 
    def _validate_shape(name, value, expected_size):
        """
        Validates the shape of the parameter value: vector or scalar? If the value 
        is a single element numpy array for a scalar parameter, it converts it to a float. 
        """
        
        # Vector paramater 
        if isinstance(expected_size, tuple): 
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Parameter '{name}' should be a numpy array.")
            if value.shape != expected_size: 
                raise ValueError(
                    f"Parameter '{name}' has incorrect shape. Expected {expected_size} but got {value.shape}."
                    )   
        
        # Scalar paramater 
        elif isinstance(expected_size, int):  
            if isinstance(value, np.ndarray):
                if value.shape == (1,): 
                    # Convert single-element numpy array to scalar
                    value = float(value)
                else:
                    raise TypeError(
                        f"Parameter '{name}' should be a scalar but got an array with shape {value.shape}."
                        )
        return value 
    
    @staticmethod 
    def _validate_constraints(name, value, constraints):
        """
        Validates constraints:
            - Max?
            - Min? 
            - PSD?
            - Simplex?
        """
        if "min" in constraints: 
            if isinstance(value, (int, float)):
                if value < constraints["min"]:
                    raise ValueError(
                        f"Parameter '{name}' must be at least {constraints['min']}. Got {value}."
                        )
            else: 
                if np.any(np.asarray(value) < constraints["min"]):
                    raise ValueError(
                        f"Parameter '{name}' must be at least {constraints['min']}. Got {value}."
                        )
                    
        if "max" in constraints:
            if isinstance(value, (int, float)):
                if value > constraints["max"]:
                    raise ValueError(
                        f"Parameter '{name}' must be at most {constraints['max']}. Got {value}."
                    )
            else:
                if np.any(np.asarray(value) > constraints["max"]):
                    raise ValueError(
                        f"Parameter '{name}' must be at most {constraints['max']}. Got {value}."
                    )
        
    def _validate_values(self, candidate_values):
        """
        Validate that the given values match the parameter group names and constraints.

        Parameters
        ----------
        candidate_values : dict
            Dictionary of parameter values to be validated before assignment.
        """
        expected_names = set(self.param.param_info.keys())
        candidate_names = set(candidate_values.keys()) # Names in user-provided dict
        
        self._validate_keys(expected_names, candidate_names) 
        
        # Iterate through parameter values and validate them
        for name, value in candidate_values.items():
            
            # Validate names
            if name not in self.param.param_info:
                raise KeyError(f"Parameter '{name}' metadata not found.")

            param_meta = self.param.param_info[name] # Extract metadata of parameter

            # Validate type
            expected_type = param_meta["type"]
            self._validate_type(name, value, expected_type)

            # Validate shape and possibly convert single-element arrays to scalars
            expected_size = param_meta["size"]
            value = self._validate_shape(name, value, expected_size)

            # Validate constraints
            constraints = param_meta.get("constraint", {})
            self._validate_constraints(name, value, constraints)

            # Update the candidate value in case it was modified (e.g., conversion from array to scalar)
            candidate_values[name] = value
                 
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_values):
        """
        Validate and update parameter values.

        Parameters
        ----------
        new_values : dict
            Dictionary containing new parameter values to be set.
        """
        self._validate_values(new_values)  # Validate before setting
        self._value = new_values  # Update internal storage

    @value.deleter
    def value(self):
        del self._value

    def to_array(self):
        raise NotImplementedError()
