import pytest

from mirrowrs.tools import FileExtensionError
from mirrowrs.tools import DisjointBboxError
from mirrowrs.tools import DimensionError

# Test if it is a subclass of TypeError
def test_fileextentionerror_is_subclass():
    assert issubclass(FileExtensionError, TypeError)

# Test if exception can be raised correctly
def test_fileextentionerror_can_be_raised_and_caught():
    with pytest.raises(FileExtensionError) as exc_info:
        raise FileExtensionError
    assert isinstance(exc_info.value, TypeError)

# Test error message
def test_fileextentionerror_error_message():
    err = FileExtensionError("Oops!")
    assert str(err) == "Oops!"

# Test if it is a subclass of ValueError
def test_disjointbboxerror_is_subclass():
    assert issubclass(DisjointBboxError, ValueError)

# Test if exception can be raised correctly
def test_disjointbboxerror_can_be_raised_and_caught():
    with pytest.raises(DisjointBboxError) as exc_info:
        raise DisjointBboxError
    assert isinstance(exc_info.value, ValueError)

# Test error message
def test_disjointbboxerror_error_message():
    err = DisjointBboxError("Oops!")
    assert str(err) == "Oops!"

# Test if it is a subclass of ValueError
def test_dimensionerror_is_subclass():
    assert issubclass(DimensionError, ValueError)

# Test if exception can be raised correctly
def test_dimensionerror_can_be_raised_and_caught():
    with pytest.raises(DimensionError) as exc_info:
        raise DimensionError
    assert isinstance(exc_info.value, ValueError)

# Test error message
def test_dimensionerror_error_message():
    err = DimensionError("Oops!")
    assert str(err) == "Oops!"