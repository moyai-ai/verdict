from dataclasses import dataclass, field

from verdict import Block, Layer, Pipeline
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.schema import Schema
from verdict.transform import MaxPoolUnit
from verdict.util import ratelimit

ratelimit.disable()

# Default prompts
DEFAULT_JUDGE_PROMPT = """
Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that
all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent.
Document: {source.doc}
Claim: {source.claim}
Please assess the claim's consistency with the document by responding with either "yes" or "no".
Answer:
"""

DEFAULT_VERIFY_PROMPT = """
Check if the given answer correctly reflects whether the claim is consistent with the corresponding document. Consistency in this context implies that
all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. "yes" means the claim is consistent with the document, and "no" means the claim is not consistent with the document.

Document: {source.doc}
Claim: {source.claim}
Answer: {previous.choice}
Answer Justification: {previous.explanation}

If you think the answer is correct, return the answer as is. If it's incorrect, return the opposite (if answer = "yes", return "no", and if answer = "no", return "yes").
"""


@dataclass
class EnsembleVerifyJudge:
    judge_prompt: str = DEFAULT_JUDGE_PROMPT
    verify_prompt: str = DEFAULT_VERIFY_PROMPT
    output_categories: DiscreteScale = field(
        default_factory=lambda: DiscreteScale(["yes", "no"])
    )
    model: str = "gpt-4o"
    repeat: int = 3
    retries: int = 3
    judge_temperature: float = 0.7
    verify_temperature: float = 0.0
    judge_explanation: bool = True
    pipeline: Pipeline = field(init=False)
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hierarchical_protocol = (
            Block()
            >> CategoricalJudgeUnit(
                name="judge",
                categories=self.output_categories,
                explanation=self.judge_explanation,
            )
            .prompt(self.judge_prompt)
            .via(
                policy_or_name=self.model,
                retries=self.retries,
                temperature=self.judge_temperature,
                **self.kwargs,
            )
            >> CategoricalJudgeUnit(name="verify", categories=self.output_categories)
            .prompt(self.verify_prompt)
            .via(
                policy_or_name=self.model,
                retries=self.retries,
                temperature=self.verify_temperature,
                **self.kwargs,
            )
        )

        self.pipeline = (
            Pipeline()
            >> Layer([self.hierarchical_protocol], repeat=self.repeat)
            >> MaxPoolUnit()
        )

    def run(self, dataset: list[Schema]) -> list[str]:
        results_df, leaf_node_cols = self.pipeline.run_from_list(dataset)
        return list(results_df[leaf_node_cols[0]])


if __name__ == "__main__":
    judge = EnsembleVerifyJudge()
    test = judge.run(
        [
            # Examples from ExpertQA
            Schema.of(
                doc="DX Bash learning on the different ways on how they can make the child comfortable and relaxed. Find non-verbal cues to connect Communicating with a child with autism may be daunting. However, you need to talk or even touch to bond and communicate. The tone of your voice and body language have a significant impact on how you communicate with a child suffering from autism; hence you need to learn their language to help improve communications between you, this will make them feel loved and accepted within the family. Maintain a personalized autism treatment plan With many different treatments available, it can",
                claim="The child can feel loved and accepted within the family this way.",
            ),
            Schema.of(
                doc="Greece History and Culture | WorkingAbroad Greece - History and Culture Greek history stretches right back to the Paleolithic era, circa 400,000 - 13,000 BP. Other periods into which the country's history is usually sorted are Mesolithic (circa 10,000 - 7000 BCE, Neolithic (circa 7000 - 3000 BCE), Bronze Age (circa 3300 - 1150 BCE), Dark Ages (circa 1100 - 700 BCE), Archaic (circa 700 - 480 BCE), Classical (480 - 323 BCE) and Hellenistic (323 - 30 BCE). And these are all prior to the infamous Roman conquest of the region circa 146 BC-324 AD. After this, the area was still influenced by the rule",
                claim="However, it is important to note that the idea of hero cults and hero worship was a common practice earlier in Greek history, dating back to the Archaic period (circa 800-480 BCE).",
            ),
        ]
    )
    print(test)
