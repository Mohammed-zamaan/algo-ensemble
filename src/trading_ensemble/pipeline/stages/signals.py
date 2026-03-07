from ..engine import PipelineStage


class SignalsStage(PipelineStage):

    name = "signals"

    def run(self, context):

        print("Generating signals")

        context["signals"] = []