import statistics
from collections.abc import Callable
from operator import attrgetter
from typing import Any

from verdict.core.primitive import Unit
from verdict.schema import Schema
from verdict.util.exceptions import VerdictExecutionTimeError
from verdict.util.misc import lightweight
from verdict.util.tracing import ExecutionContext


@lightweight
class MapUnit(Unit):
    _char: str = "Map"

    accumulate: bool = True
    map_func: Callable[[Any | list[Any]], Any | list[Any]]

    def __init__(
        self,
        map_func: Callable[[Any | list[Any]], Any | list[Any]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data.map_func = lambda x: map_func(x)

    class InputSchema(Schema):
        values: Any | list[Any]

    class ResponseSchema(Schema):
        values: Any | list[Any]

    def execute(
        self, input: InputSchema, execution_context: ExecutionContext | None = None
    ) -> ResponseSchema:
        if execution_context is None:
            execution_context = ExecutionContext()

        call_name = (
            self.__class__.__name__
            or getattr(self, "_char", None)
            or getattr(self, "char", None)
        )

        with execution_context.trace_call(
            name=call_name,
            inputs={"input": input, "unit": self},
        ) as call:
            try:
                values = input.values
                if len(values) == 1:
                    values = values[0]

                output = self.data.map_func(values)
                if isinstance(output, Schema):
                    self.OutputSchema = self.ResponseSchema = output.__class__  # type: ignore
                    result = output  # type: ignore
                else:
                    result = MapUnit.ResponseSchema(values=output)

                if call is not None:
                    call.set_outputs(result)
                return result
            except Exception as e:
                raise VerdictExecutionTimeError(
                    f"Failed to execute MapUnit: {e} on {input}"
                ) from e


class FieldMapUnit(MapUnit):
    fields: list[str]

    def __init__(
        self,
        map_func: Callable[[Any | list[Any]], Any | list[Any]],
        fields: str | list[str] | None,
        **kwargs,
    ):
        super().__init__(map_func, **kwargs)
        if fields is None:
            fields = []
        self.fields = fields if isinstance(fields, list) else [fields]

    def execute(
        self,
        input: MapUnit.InputSchema,
        execution_context: ExecutionContext | None = None,
    ) -> MapUnit.ResponseSchema:
        if execution_context is None:
            execution_context = ExecutionContext()

        call_name = (
            self.__class__.__name__
            or getattr(self, "_char", None)
            or getattr(self, "char", None)
        )

        with execution_context.trace_call(
            name=call_name,
            inputs={"input": input, "unit": self},
        ) as call:
            values = input.values

            if len(self.fields) == 0 and len(values) > 0:
                self.fields = list(values[0].model_fields.keys())

            assert all(field in values[0].model_fields for field in self.fields), (
                f"Fields {self.fields} not a subset of input {input.values}"
            )

            result = Schema.of(
                **{  # type: ignore
                    field: self.data.map_func(list(map(attrgetter(field), values)))
                    for field in self.fields
                }
            )

            if call is not None:
                call.set_outputs(result)
            return result

    @staticmethod
    def from_fn(
        fn: Callable[[list[Any]], Any], name: str
    ) -> Callable[[str | list[str] | None], "FieldMapUnit"]:
        return lambda fields=None: FieldMapUnit(fn, fields, name=name)


class MeanPoolUnit(FieldMapUnit):
    def __init__(self, fields: str | list[str] | None = None):
        super().__init__(statistics.mean, fields, name="MeanPool")


class MaxPoolUnit(FieldMapUnit):
    def __init__(self, fields: str | list[str] | None = None):
        super().__init__(statistics.mode, fields, name="MaxPool")


class MeanVariancePoolUnit(FieldMapUnit):
    def __init__(self, fields: str | list[str] | None = None):
        super().__init__(
            lambda x: Schema.inline(mean=float, variance=float)(  # type: ignore
                mean=statistics.mean(x),
                variance=statistics.variance(x) if len(x) > 1 else 0.0,
            ),
            fields,
            name="MeanVariancePool",
        )


__all__ = [
    "MapUnit",
    "MaxPoolUnit",
    "MeanPoolUnit",
    "MeanVariancePoolUnit",
]
