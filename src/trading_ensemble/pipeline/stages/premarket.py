from ..engine import PipelineStage


class PremarketStage(PipelineStage):

    name = "premarket"

    def run(self, context):

        print("Running premarket data preparation")

        context["premarket_ready"] = True