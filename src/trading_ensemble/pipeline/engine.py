from datetime import datetime
from typing import List


class PipelineStage:
    name: str

    def run(self, context: dict) -> None:
        raise NotImplementedError


class PipelineEngine:
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, context: dict | None = None):
        context = context or {}

        print("=" * 60)
        print("TRADING PIPELINE START")
        print(datetime.utcnow())
        print("=" * 60)

        for stage in self.stages:
            print(f"\n--- Running stage: {stage.name} ---")
            stage.run(context)

        print("\nPipeline finished")
        return context