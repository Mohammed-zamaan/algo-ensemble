from ..engine import PipelineStage


class EliminationStage(PipelineStage):

    name = "elimination"

    def run(self, context):

        print("Running candidate elimination")

        context["candidates"] = []