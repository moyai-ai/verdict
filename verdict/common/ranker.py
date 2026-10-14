import string

from verdict.core.primitive import Unit
from verdict.scale import DiscreteScale
from verdict.schema import Field, Schema


class RankerUnit(Unit):
    _char: str = "Ranker"

    class InputSchema(Schema):
        # need a sentinel for empty, but using [''] for now makes type inference simple
        options: list[str] = Field(default=[""])

    class ResponseSchema(Schema):
        ranking: list[str]

    class OutputSchema(Schema):
        ranked_options: list[str]

    def __init__(
        self,
        k: int = 3,
        options: DiscreteScale | None = None,
        explanation: bool = False,
        original: bool = False,
        **kwargs,
    ):
        self.k = k
        self.explanation = explanation
        self.original = original

        options = options or DiscreteScale(list(string.ascii_uppercase[:k]))
        self.scale = options

        if explanation:
            self.ResponseSchema = self.ResponseSchema.prepend(explanation=str)  # type: ignore
            self.OutputSchema = self.OutputSchema.prepend(explanation=str)  # type: ignore

        super().__init__(**kwargs)

    def validate(self, input: InputSchema, response: ResponseSchema) -> None:
        if self.original:
            assert input.options != [""], (
                "Pass the options in InputSchema.options to index original choices."
            )
            assert len(input.options) == self.k, (
                f"Number of input options must equal {self.k}"
            )

    def process(self, input: InputSchema, response: ResponseSchema) -> Schema:
        if not self.original:
            return response

        fields = {
            "ranked_options": [
                input.options[
                    self.scale.values.index(self.scale.value_mapping_fn(rank))
                ]
                for rank in response.ranking
            ],
        }

        if self.explanation:
            fields["explanation"] = response.explanation  # type: ignore[attr-defined]

        return self.OutputSchema(**fields)
