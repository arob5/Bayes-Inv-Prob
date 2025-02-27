#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:39 2025

@author: vigneshsomjit
"""
import numpy as np
import copy
import math


class _NoConstraintType(object):
    """
    Defines a second "None" type that represents the absense of a parameter
    constraint. This is done as opposed to using None for this purpose, as we
    want to reserve None to indicate the absence of a function argument.
    There is nothing restricting this class to be a singleton, but only a single
    "NoConstraint" object should be created to be able to treat the object
    as one does "None". This code was borrowed from the following StackOverflow
    post:
    https://stackoverflow.com/questions/41048643/how-to-create-a-second-none-in-
    python-making-a-singleton-object-where-the-id-is
    """
    def __new__(cls):
        return NoConstraint
    def __reduce__(self):
        return (_NoConstraintType, ())

NoConstraint = object.__new__(_NoConstraintType)


class ParamInfo:
    """
    Encapsulates metadata for a parameter.
    """
    def __init__(self,
                 value_type: str,
                 shape: tuple[()] | tuple[int, ...], # TODO: and this one
                 constraint: str | tuple[int|None,int|None]): # TODO: need to check this type hint is correct.
        """

        Parameters
        ----------
        value_type : `str`
            The Python type of the parameter. For array-valued parameters, this
            is the type of the individual entries of the array. Allows either
            "float" or "int".
        shape : `tuple`
            The shape of the parameter, following NumPy shape conventions. A
            scalar parameter has shape ().
        constraint : `str`, `tuple` of length 2, or NoConstraint.
            Parameter constraint (in addition to the type constraint); only one
            constraint allowed. Options:
            - NoConstraint: will not impose additional constraint.
            - (lower, upper): imposes lower and upper bound constraints.
              Interpreted elementwise for array-valued parameters. One-sided
              bounds can be provided by setting one of the bounds to None.
            - "psd": only valid shape implies a square matrix. Enforces
              positive semidefiniteness.
            - "simplex": Enforces constraint that the entries of the parameter
              must be in [0,1] and sum to one.
        """

        # Initialize to None so that call to `_ensure_consistent_info` works.
        self._value_type = None
        self._shape = None
        self._constraint = None

        # Call setters, which validate parameter info.
        self.value_type = value_type
        self.shape = shape
        self.constraint = constraint

    @property
    def value_type(self):
        return self._value_type

    @property
    def shape(self):
        return self._shape

    @property
    def constraint(self):
        return self._constraint

    @value_type.setter
    def value_type(self, value_type):
        ParamInfo._validate_value_type(value_type)
        self._ensure_consistent_info(value_type=value_type)
        self._value_type = value_type

    @shape.setter
    def shape(self, shape):
        ParamInfo._validate_shape(shape)
        self._ensure_consistent_info(shape=shape)
        self._shape = shape

    @constraint.setter
    def constraint(self, constraint):
        constraint = ParamInfo._validate_constraint(constraint)
        self._ensure_consistent_info(constraint=constraint)
        self._constraint = constraint

    @staticmethod
    def _validate_value_type(value_type):
        if not isinstance(value_type, str):
            raise TypeError(f"value_type must be a string, not {type(value_type)}.")

        if value_type not in ("float", "int"):
            raise ValueError(f"value_type must equal 'float' or 'int', got {value_type}.")

    @staticmethod
    def _validate_shape(shape):
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple, not {type(shape)}.")

        if not all([isinstance(x, int) for x in shape]):
            raise TypeError("shape tuple must only contain integers.")

    @staticmethod
    def _validate_constraint(constraint):
        if not isinstance(constraint, (str, tuple)) and constraint is not NoConstraint:
            raise TypeError("constraint must either be NoConstraint, a string,"
                            f"or a tuple, not {type(constraint)}.")

        if isinstance(constraint, tuple): # Bound constraint
            constraint = ParamInfo._validate_bound_constraint(constraint)
        elif isinstance(constraint, str): # Non-bound constraint
            if not constraint in ("psd", "simplex"):
                raise ValueError("If constraint is string, must be 'psd' or"
                                 f" 'simplex', got {constraint}.")

        return constraint

    def _ensure_consistent_info(self, value_type=None, shape=None, constraint=None):
        # TODO: need to grab the current values of any params that aren't set.
        # But also need to handle the case that they haven't been set yet.

        # If only on piece of info is being changed, select the current values
        # for the other pieces of info.
        if value_type is None:
            value_type = self.value_type
        if shape is None:
            shape = self.shape
        if constraint is None:
            constraint = self.constraint

        # Validate dependencies across type, shape, constraint.
        if constraint == "psd":
            if not self.is_square_matrix(shape):
                raise ValueError("Constraint 'psd' only valid for square matrix.")

        if constraint == "simplex":
            if value_type != "float":
                raise ValueError("Constraint 'simplex' requires 'float' value_type.")

    @staticmethod
    def _validate_bound_constraint(constraint):

        if len(constraint) != 2:
            raise TypeError("If constraint is a tuple, must be length 2 "
                            f"containing lower and upper bounds. Got {constraint}.")

        if not all([isinstance(x, (int,float)) or x is None for x in constraint]):
            raise TypeError("If constraint is a tuple, elements must be either"
                            "None, or of type Int or Float.")

        lower, upper = constraint

        if (lower is not None) and (upper is not None):
            if lower > upper:
                raise ValueError(f"Invalid bound constraint ({lower}, {upper})."
                                  " Lower bound exceeds upper bound.")

        # Convert infinite bounds to None.
        if (lower is not None) and math.isinf(lower):
            lower = None
        if (upper is not None) and math.isinf(upper):
            upper = None
        bounds = (lower, upper)

        # If lower and upper are both None, simplify by setting to NoConstraint.
        if (lower is None) and (upper is None):
            bounds = NoConstraint

        return bounds


    def __str__(self):
        constraint = self.constraint
        if constraint is NoConstraint:
            constraint = "NoConstraint"

        return(f"value_type: {self.value_type}\n"
               f"shape: {self.shape}\n"
               f"constraint: {constraint}\n"
               f"length: {len(self)}")

    def __len__(self):
        if len(self.shape) == 0: # Empty tuple, implies scalar parameter.
            return 1

        return np.prod(self.shape)

    @staticmethod
    def is_matrix(shape):
        return len(shape) == 2

    @staticmethod
    def is_square_matrix(shape):
        if not ParamInfo.is_matrix(shape):
            return False

        return shape[0] == shape[1]


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
