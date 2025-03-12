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
    This class encapsulates metadata for a parameter, which includes its type,
    shape, and constraints. It ensures that parameters conform to specified rules
    and enforces cross-parameter consistency. The class attributes encoding
    the parameter information are regarded as immutable; they should be set
    when instantiating the class and not changed thereafter.

    `ParamInfo` objects are used when creating an instance of the `ParamGroup`
    class.
    """
    def __init__(self,
                 value_type: str,
                 shape: tuple[()] | tuple[int, ...],
                 constraint: str | tuple[int | None, int | None]):
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

        # Set parameter info.
        self._value_type = value_type
        self._shape = shape
        self._constraint = constraint

        # Validate, and potentially simplify, parameter info.
        self._validate_info()
        self._simplify_info()

    @property
    def value_type(self):
        return self._value_type

    @property
    def shape(self):
        return self._shape

    @property
    def constraint(self):
        return self._constraint

    def _validate_info(self):
        self._validate_value_type()
        self._validate_shape()
        self._validate_constraint()
        self._ensure_consistent_info()

    def _validate_value_type(self):
        value_type = self.value_type
        if not isinstance(value_type, str):
            raise TypeError(f"value_type must be a string, not {type(value_type)}.")

        if value_type not in ("float", "int"):
            raise ValueError(f"value_type must equal 'float' or 'int', got {value_type}.")

    def _validate_shape(self):
        shape = self.shape
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple, not {type(shape)}.")

        if not all([isinstance(x, int) for x in shape]):
            raise TypeError("shape tuple must only contain integers.")

    def _validate_constraint(self):
        constraint = self.constraint
        if not isinstance(constraint, (str, tuple)) and constraint is not NoConstraint:
            raise TypeError("constraint must either be NoConstraint, a string,"
                            f" or a tuple, not {type(constraint)}.")

        if ParamInfo.is_bound_constraint(constraint): # Bound constraint
            constraint = self._validate_bound_constraint()
        elif isinstance(constraint, str): # Non-bound constraint
            if not constraint in ("psd", "simplex"):
                raise ValueError("If constraint is string, must be 'psd' or"
                                 f" 'simplex', got {constraint}.")

    def _ensure_consistent_info(self):
        value_type = self.value_type
        shape = self.shape
        constraint = self.constraint

        # Validate dependencies across type, shape, constraint.
        if constraint == "psd":
            if not ParamInfo.is_square_matrix(shape):
                raise ValueError("Constraint 'psd' only valid for square matrix.")

        if constraint == "simplex":
            if value_type != "float":
                raise ValueError("Constraint 'simplex' requires 'float' value_type.")

    def _validate_bound_constraint(self):
        constraint = self.constraint
        if not ParamInfo.is_bound_constraint(constraint):
            return None

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

    @staticmethod
    def is_bound_constraint(constraint):
        return isinstance(constraint, tuple)

    def _simplify_info(self):
        # At present, simplfications can only be made for bound constraints.

        constraint = self.constraint
        if not ParamInfo.is_bound_constraint(constraint):
            return None

        lower, upper = constraint

        # Convert infinite bounds to None.
        if (lower is not None) and math.isinf(lower):
            lower = None
        if (upper is not None) and math.isinf(upper):
            upper = None
        bounds = (lower, upper)

        # If lower and upper are both None, simplify by setting to NoConstraint.
        if (lower is None) and (upper is None):
            bounds = NoConstraint

        # Update constraint attribute.
        self._constraint = bounds

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


    def __eq__(self, other):
        """
        Overloads the '==' operator to test for equality between two ParamInfo objects.

        Two ParamInfo objects are considered equal if:
            1) They have the same `value_type`.
            2) They have the same `shape`.
            3) They have the same `constraint`.

        Parameters
        ----------
        other : `ParamInfo`
            Another ParamInfo instance to compare against.

        """
        if not isinstance(other, ParamInfo):
            return NotImplemented  # Return False if not ParamInfo object

        return (self.value_type == other.value_type and
                self.shape == other.shape and
                self.constraint == other.constraint)


class ParamGroup:
    """
    This is a container class that holds multiple parameter name-metadata pairs
    and provides functionality to view, add, and delete parameters.

    """

    def __init__(self, param_group: dict[str, ParamInfo]):
        """
        Parameters
        ----------
        params : dict
            Dictionary storing the name-metadata pairing of all the parameters
            in the group. The keys are the parameter names and the values
            are `ParamInfo` objects.

        Raises
        ------
        TypeError
            If `param_group` is not a dictionary with string keys and `ParamInfo` values.
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dictionary.")

        for key, value in param_group.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Invalid key '{key}': keys must be strings, got {type(key)}."
                )
            if not isinstance(value, ParamInfo):
                raise TypeError(
                    f"Invalid value for '{key}': must be a ParamInfo object, got {type(value)}."
                )

        self._param_group = param_group

    def get_param_names(self, flatten = False) -> list[str]:
        """
        Return a list of parameter names in alphabetical order.

        Parameters
        ----------
        flatten : bool, optional
            If False, then list consists of the set of keys in `self._param_group`.
            If True, then the individual elements of array-valued parameters are included.

        Returns
        -------
        list[str]
            A sorted list of parameter names.

        """
        param_names = sorted(list(self._param_group.keys()))
        if not flatten:
            return param_names

        names_flat = []
        for name in param_names:
            info = self._param_group[name]
            if len(info) == 1:
                names_flat.append(name)
            else:
                subnames = ParamGroup.construct_arr_elem_names(info.shape, flatten=True)
                names_flat += np.char.add(name, subnames).tolist()

        return names_flat

    def add_param(self, param_name: str, param_metadata: ParamInfo) -> None:
        """
        Add a single new parameter to the group.

        Parameters
        ----------
        param_name : str
            The new parameter name.
        param_metadata : ParamInfo
            The parameter information object.

        Raises
        ------
        TypeError
            If `param_name` is not a string or `param_metadata` is not a `ParamInfo` instance.
        KeyError
            If the parameter already exists in the group.
        """
        if not isinstance(param_name, str):
            raise TypeError("param_name must be a string.")
        if not isinstance(param_metadata, ParamInfo):
            raise TypeError("param_metadata must be an instance of ParamInfo.")
        if param_name in self._param_group:
            raise KeyError(f"Parameter '{param_name}' already exists in the group.")

        self._param_group[param_name] = param_metadata

    def remove_param(self, param_names: str | list[str] | tuple[str]) -> None:
        """
        Remove one or more parameters from the group by name.

        Parameters
        ----------
        param_names : `str`, `list`, or `tuple`
            The parameter name(s) to remove.

        Raises
        ------
        TypeError
            If `param_names` is not a string, list, or tuple.
        KeyError
            If any parameter in `param_names` does not exist.
        """
        if isinstance(param_names, str):
            param_names = [param_names]

        if not isinstance(param_names, (list, tuple)):
            raise TypeError("param_names must be a string, list, or tuple.")

        for param_name in param_names:
            if param_name not in self._param_group:
                raise KeyError(f"Parameter '{param_name}' does not exist in the group.")
            self._param_group.pop(param_name)

    @staticmethod
    def construct_arr_elem_names(shape: tuple[int], flatten: bool = False):
        """
        For a given array shape, constructs string names for each entry, where
        the names are of the form of the expression that would be used to
        select the entry when indexing the array; e.g., the (1,4,2) element of
        a 3-dim array is given the name "[1,4,2]".

        Examples
        -----
        construct_arr_elem_names((3,)) returns ["[0]", "[1]", "[2]"].
        construct_arr_elem_names((2,2), flatten=True) returns
        ["[0,0]", "[0,1]", "[1,0]", "[1,1]"].
        construct_arr_elem_names(()) returns [].

        Parameters
        ----------
        shape : `tuple`
            The shape of the array.
        flatten : `bool`
            If False, returned array has shape `shape`. Otherwise, returned
            array is flattened using `np.flatten`.

        Returns
        -------
        np.ndarray[str]
            The constructed names. See `flatten` above for the return
            array dimension.
        """
        idcs = np.indices(shape).astype("str")
        idx_names = np.char.add("[", idcs[0])

        for i in range(1, idcs.shape[0]):
            idx_names = np.char.add(idx_names, ",")
            idx_names = np.char.add(idx_names, idcs[i])

        idx_names = np.char.add(idx_names, "]")

        if flatten:
            return idx_names.flatten(order="C")
        return(idx_names)

    def __len__(self) -> int:
        """
        Returns the total number of individual scalar values across all
        parameters in the group. Calculated as sum(prod(shape of each parameter)).

        Notes
        -----
        This method relies on `ParamInfo.__len__()` which calculates the
        number of elements based on the shape of the parameter.

        Returns
        -------
        int
            The total number of scalar elements across all parameters.
        """
        return sum(len(param_info) for param_info in self._param_group.values())

    def __str__(self) -> str:
        """
        Returns a tabular string representation of the ParamGroup.

        The table includes the following columns:
            1) Parameter Name
            2) Value Type
            3) Shape
            4) Constraint

        Returns
        -------
        str
            A formatted table string representing the parameter group.
        """
        if not self._param_group:
            return "ParamGroup is empty."

        # Define table headers
        headers: list[str] = ["Parameter Name", "Type", "Shape", "Constraint"]

        # Collect row data
        rows:list[list[str]] = []
        for name, param in self._param_group.items():
            constraint = param.constraint
            if constraint is NoConstraint:
                constraint = "NoConstraint" # For nicer print format.
            rows.append([name, param.value_type, param.shape, constraint])

        # Determine column widths
        col_widths: list[int] = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]

        # Format header row
        header_row = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        separator = "-|-".join("-" * col_widths[i] for i in range(len(headers)))

        # Format data rows
        formatted_rows = [" | ".join(f"{str(row[i]):<{col_widths[i]}}"
                                     for i in range(len(headers))) for row in rows]

        # Combine into final table format
        return "\n".join([header_row, separator] + formatted_rows)

    def __eq__(self, other) -> bool:
        """
        Overloads the '==' operator to test for equality between two ParamGroup objects.

        Two ParamGroup instances are considered equal if:
            1) They have the same set of parameter names.
            2) Each parameter has the same value type, shape, and constraint.

        Parameters
        ----------
        other : object
            Another ParamGroup instance to compare against.

        Returns
        -------
        bool
            True if both ParamGroup instances are identical in structure and values, False otherwise.
        """
        if not isinstance(other, ParamGroup):
            return NotImplemented  # Returns False if not ParamGroup

        # Check if they have the same set of parameter names
        if set(self.get_param_names()) != set(other.get_param_names()):
            return False

        # Check if all corresponding ParamInfo objects are identical
        for key in self.get_param_names():
            if self._param_group[key] != other._param_group[key]:
                return False

        return True


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
        self.param_group = param
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
    def _validate_shape(name, value, expected_shape):
        """
        Validates the shape of the parameter value: vector or scalar? If the value
        is a single element numpy array for a scalar parameter, it converts it to a float.
        """

        # Vector paramater
        if isinstance(expected_shape, tuple):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Parameter '{name}' should be a numpy array.")
            if value.shape != expected_shape:
                raise ValueError(
                    f"Parameter '{name}' has incorrect shape. Expected {expected_shape} but got {value.shape}."
                    )

        # Scalar paramater
        elif isinstance(expected_shape, int):
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
        expected_names = set(self.param_group.param_info.keys())
        candidate_names = set(candidate_values.keys()) # Names in user-provided dict

        self._validate_keys(expected_names, candidate_names)

        # Iterate through parameter values and validate them
        for name, value in candidate_values.items():

            # Validate names
            if name not in self.param_group.param_info:
                raise KeyError(f"Parameter '{name}' metadata not found.")

            param_meta = self.param_group.param_info[name] # Extract metadata of parameter

            # Validate type
            expected_type = param_meta["type"]
            self._validate_type(name, value, expected_type)

            # Validate shape and possibly convert single-element arrays to scalars
            expected_shape = param_meta["shape"]
            value = self._validate_shape(name, value, expected_shape)

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
