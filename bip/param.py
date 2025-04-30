#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:39 2025

@author: vigneshsomjit
"""
import numpy as np
import math

def sort_mixed_types(l):
    # Sorts a `list` or `dict_keys` that potentially contains mixed types
    # (e.g, str and int). In the str/int case, the sorted integer keys will
    # be placed first, followd by the sorted string keys.
    return sorted(l, key=lambda x: (type(x).__name__, x))

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
        """
        Validates the parameter metadata by checking type, shape, constraint,
        and consistency across these attributes.
        """
        self._validate_type()
        self._validate_shape()
        self._validate_constraint()
        self._ensure_consistent_info()

    def validate_value(self, val):
        """
        Validates that a given NumPy array 'val' satisfies the type, shape,
        and constraint specified by this ParamInfo.
        """
        val = np.asarray(val)

        if not isinstance(val, np.ndarray):
            raise TypeError(f"Value must be a numpy.ndarray, got {type(val)}")

        self._validate_value_type(val)
        self._validate_value_shape(val)
        self._validate_value_constraint(val)

    def _validate_type(self):
        """"
        Checks that the value_type attribute is a string and is either 'float'
        or `int`.
        """
        value_type = self.value_type
        if not isinstance(value_type, str):
            raise TypeError(f"value_type must be a string, not {type(value_type)}.")

        if value_type not in ("float", "int"):
            raise ValueError(f"value_type must equal 'float' or 'int', got {value_type}.")

    def _validate_shape(self):
        """
        Ensures that the shape attribute is a tuple containing only integers.
        """
        shape = self.shape
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple, not {type(shape)}.")

        if not all([isinstance(x, int) for x in shape]):
            raise TypeError("shape tuple must only contain integers.")

    def _validate_constraint(self):
        """
        Validates that the constraint attribute is either:
            - NoConstraint,
            - a tuple (bound constraint),
            - or a string ("psd" or "simplex").
        """
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
        """
        Ensures consistency between the type, shape and constraint.
        For example, the 'psd' constraint is only valid for square matrices,
        and 'simpex' requires float values.
        """
        value_type = self.value_type
        shape = self.shape
        constraint = self.constraint

        # Validate dependencies across type, shape, constraint.
        if constraint == "psd":
            if not ParamInfo.is_square_matrix_shape(shape):
                raise ValueError("Constraint 'psd' only valid for square matrix.")

        if constraint == "simplex":
            if value_type != "float":
                raise ValueError("Constraint 'simplex' requires 'float' value_type.")

    def _validate_bound_constraint(self):
        """
        Validates a bound constraint tuple, ensuring it has two elements
        that are numbers or None. Also checks that if both bounds are provided,
        the lower bound is not greater than the upper bound.
        """
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

    def _validate_value_type(self, val):
        """
        Validates that the Numpy array 'val' has the correct data type as
        specified in value_type.
        """
        required_type = self.value_type

        if required_type == "float":
            if not np.issubdtype(val.dtype, np.floating):
                raise TypeError(f"Value must have numpy.floating dtype, not {val.dtype}")

        if required_type == "int":
            if not np.issubdtype(val.dtype, np.signedinteger):
                raise TypeError(f"Value must have numpy.signedinteger dtype, not {val.dtype}")

    def _validate_value_shape(self, val):
        """
        Validates the shape of the Numpy array 'val' matches the expected shape.
        """
        required_shape = self.shape

        if val.shape != required_shape:
            raise TypeError(f"Param has shape {required_shape}, but value has shape {val.shape}")

    def _validate_value_constraint(self, val):
        """
        Validates that the Numpy array 'val' satisfies the specified constraint.
        """
        constraint = self.constraint
        if ParamInfo.is_bound_constraint(constraint):
            self._check_value_satisfies_bounds(val)
        elif constraint == "psd":
            ParamInfo.check_value_is_psd(val)
        elif constraint == "simplex":
            ParamInfo.check_value_in_simplex(val)

    def _check_value_satisfies_bounds(self, val):
        """
        For a bound constraint, checks each element of the Numpy array 'val' to ensure
        it does not violate the lower or upper bounds.
        """
        bounds = self.constraint
        if not ParamInfo.is_bound_constraint(bounds):
            return None

        lower, upper = bounds

        if lower is not None:
            n_lower_bound_violations = (val < lower).sum()
            if n_lower_bound_violations > 0:
                raise ValueError(f"Value has {n_lower_bound_violations}"
                                 " lower bound violation(s).")

        if upper is not None:
            n_upper_bound_violations = (val > upper).sum()
            if n_upper_bound_violations > 0:
                raise ValueError(f"Value has {n_upper_bound_violations}"
                                 " upper bound violation(s).")

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

    @staticmethod
    def check_value_is_psd(val):
        if not ParamInfo.is_psd_matrix(val):
            raise ValueError("Value violates positive semidefinite constraint.")

    @staticmethod
    def check_value_in_simplex(val):
        n_zero_violations = (val < 0.0).sum()
        if n_zero_violations > 0:
            raise ValueError(f"Array has {n_zero_violations} element(s)"
                             " less than zero.")

        n_one_violations = (val > 1.0).sum()
        if n_one_violations > 0:
            raise ValueError(f"Array has {n_one_violations} element(s)"
                             " exceeding one.")

        if not np.isclose(val.sum()-1, 0.0, atol=1e-08):
            raise ValueError("Violation of simplex sum-to-one constraint.")

    @staticmethod
    def is_bound_constraint(constraint):
        return isinstance(constraint, tuple)

    @staticmethod
    def is_matrix_shape(shape):
        return len(shape) == 2

    @staticmethod
    def is_square_matrix_shape(shape):
        if not ParamInfo.is_matrix_shape(shape):
            return False

        return shape[0] == shape[1]

    @staticmethod
    def is_symmetric_matrix(A):
        return np.array_equal(A, A.T)

    @staticmethod
    def is_psd_matrix(A):
        """
        Some useful sources:
        https://scicomp.stackexchange.com/questions/12979/testing-if-a-matrix-is-positive-semi-definite
        https://stackoverflow.com/questions/16266720/find-out-if-a-matrix-is-positive-definite-with-numpy
        """

        if ParamInfo.is_symmetric_matrix(A):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def __str__(self):
        constraint = self.constraint
        if constraint is NoConstraint:
            constraint = "NoConstraint"

        return(f"ParamInfo<value_type: {self.value_type}, "
               f"shape: {self.shape}, "
               f"constraint: {constraint}>")

    def __len__(self):
        if len(self.shape) == 0: # Empty tuple, implies scalar parameter.
            return 1

        return np.prod(self.shape)

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

    @property
    def param_group(self):
        return self._param_group

    def get_param_names(self, flatten=False) -> list[str]:
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


class ParamGroupValues:
    """
    Encapsulates the actual value(s) that a parameter group (encoded by a
    ParamGroup object) assumes. Includes validation to ensure the value aligns
    with the type, shape, and constraints included in the ParamGroup. Can hold
    multiple values at once.
    """
    def __init__(self, param_group, init_values=None):
        """

        Parameters
        ----------
        param : `ParamGroup`
            Instance of ParamGroup class, defining the parameter structure.
        init_values : `Dict`
            Dictionary storing the initial values. Each element of this
            dictionary is a parameter value, which is itself encoded via
            a dictionary. The inner dictionary has keys corresponding to
            parameter names, and values the associated parameter values.
        """

        if init_values == {}:
            init_values = None

        self._param_group = param_group
        self.values = init_values

    @property
    def param_group(self):
        return self._param_group

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        """
        Validate and update parameter values.

        Parameters
        ----------
        new_values : dict
            Dictionary of dictionaries containing new parameter values to be
            set. The outer dictionary is over the different parameter values,
            with no constraints on the keys. The keys will commonly be integers
            when there is a natural ordering (e.g., MCMC samples). Alternatively,
            the keys can be strings. The inner dictionaries have keys equal to
            the parameter names in the group, with values set to their associated
            values.
        """
        # Validate each provided value.
        for val in new_values.values():
            self._validate_value(val)

        # If values are valid, update internal storage.
        self._values = new_values
        
    def _validate_value(self, candidate_values):
        """
        Validate given parameter values, which means checking that (1) the
        values are given in a dictionary, with keys matching the _param_group
        param names; (2) the value is of the correct type, shape, and satisfies
        the constraints laid out by the ParamInfo objects stored in
        _param_group. Note that this method validates a single value of the
        parameter group.

        Parameters
        ----------
        candidate_values : `dict`
            Dictionary of parameter values to be validated before assignment.
        """

        # Empty value.
        if (candidate_values == {}) or (candidate_values is None):
            return None

        if not isinstance(candidate_values, dict):
            raise TypeError(f"candidate_values must be a dict, not {type(candidate_values)}.")

        self._validate_param_names(candidate_values)

        # Validate value for each parameter in the group.
        for name, val in candidate_values.items():
            self._validate_param_value(name, val)

    def _validate_param_names(self, candidate_values: dict):
        """
        Validates the names of a dictionary containing parameter group values.
        Ensures the set of dictionary keys is equal to the set of parameter
        names in _param_group.

        Parameters
        ----------
        candidate_values : `dict`
            Dictionary of parameter values; only the keys are validated here.
        """

        candidate_names = set(candidate_values.keys())
        param_names = set(self._param_group.get_param_names())

        if candidate_names != param_names:
            # Check for extra and/or missing parameters.
            missing_names = param_names - candidate_names
            extra_names = candidate_names - param_names
            raise KeyError("Parameter name mismatch:\n"
                           f"Missing names: {missing_names}\n"
                           f"Extra names: {extra_names}")

    def _validate_param_value(self, param_name, candidate_value):
        """
        Validates a candidate value of a single parameter within the param group.
        In particular, checks against the value_type, shape, and constraint
        attributes stored in the ParamInfo object for that parameter. This
        is simply a light wrapper around the `validate_value` method of
        the `ParamInfo` class.

        Parameters
        ----------
        param_name : `str`
            A parameter name (one of the names in the param group).
        candidate_value : `np.ndarray`
            A candidate value for the parameter with name `param_name`.
        """

        self._param_group.param_group[param_name].validate_value(candidate_value)

    def to_array(self, simplify=False):
        """
        Converts the stored parameter value into a single flattened 2D NumPy
        array (i.e., a matrix). The columns of the matrix are ordered to align
        with the order of the parameter names returned by
        `_param_group.get_param_names(flatten=True)`. Each row is a single
        parameter group value, and the rows are ordered by applying
        `sort_mixed_types` to the keys of the `self.values` dictionary. This
        will sort integer keys in ascending order, and string keys
        alphabetically ascending as well. A mixture of integer and string keys
        will put the sorted integer keys first, followed by the sorted
        string keys. All values are coerced to floats.

        Parameters
        ----------
        simplify : `bool`
            If True, then will simplify a one-row matrix to a 1d numpy array.
            Otherwise, will maintain the 2d structure.
        """
        # If no values are set, return an empty array.
        if self.values is None:
            return np.array([])

        # Determine row order.
        row_order = sort_mixed_types(self.values.keys())

        # Parameter order for columns. Multivariate parameters will be
        # flattened.
        param_names = self.param_group.get_param_names(flatten=False)

        # Matrix in which to store values.
        n_vals = len(row_order)
        mat = np.empty((n_vals, len(self.param_group)), dtype=float)

        for row_idx, row_key in enumerate(row_order):
            val_flat = []
            for param_name in param_names:
                param_val = np.asarray(self.values[row_key][param_name]).flatten()
                val_flat.append(param_val)
            mat[row_idx] = np.concatenate(val_flat)

        # If there is only a single value, then optionally reduce the 1 row
        # matrix to a 1d array.
        if simplify and (n_vals == 1):
            return mat[0]

        return mat
