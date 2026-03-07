from ..engine import PipelineStage


class RiskStage(PipelineStage):

    name = "risk"

    def run(self, context):

        print("Applying risk management")

        context["orders"] = []