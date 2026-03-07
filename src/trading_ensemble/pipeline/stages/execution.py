from ..engine import PipelineStage


class ExecutionStage(PipelineStage):

    name = "execution"

    def run(self, context):

        print("Executing orders")

        orders = context.get("orders", [])

        print(f"{len(orders)} orders to execute")