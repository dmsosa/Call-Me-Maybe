"""Contrat des types : ``value_matches`` et ``format_schema_error``."""
import pytest
from pydantic import BaseModel, ValidationError

from src.exception.exception import format_schema_error
from src.model.schema import FnDefinition, ParamSpec, ParamType, value_matches


def test_function_definition_rejects_a_quote_in_a_param_name() -> None:
    """Un nom de paramètre est écrit dans le JSON : il doit être sûr."""
    with pytest.raises(ValidationError):
        FnDefinition(
            name="fn_odd", parameters={'a"b': ParamSpec(type="number")}
        )


def test_function_definition_accepts_snake_case_param_names() -> None:
    fn = FnDefinition(
        name="fn_ok", parameters={"user_id": ParamSpec(type="number")}
    )

    assert list(fn.parameters) == ["user_id"]


@pytest.mark.parametrize(
    "value,declared,expected",
    [
        (1, "number", True),
        (1.5, "number", True),
        (True, "number", False),  # bool est une sous-classe de int
        ("1", "number", False),
        (1, "integer", True),
        (1.5, "integer", False),
        (True, "integer", False),
        ("x", "string", True),
        (1, "string", False),
        (True, "boolean", True),
        (1, "boolean", False),
        (None, "string", False),
    ],
)
def test_value_matches(
    value: object, declared: ParamType, expected: bool
) -> None:
    assert value_matches(value, declared) is expected


class Small(BaseModel):
    count: int


def test_format_schema_error_names_the_path_and_the_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Small.model_validate({"count": "x"})

    message = format_schema_error(excinfo.value, "data/input/x.json")

    assert "data/input/x.json" in message
    assert "count" in message
    assert "Value error" not in message
