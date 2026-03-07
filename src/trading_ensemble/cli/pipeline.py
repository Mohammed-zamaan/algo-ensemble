from trading_ensemble.pipeline.engine import PipelineEngine

from trading_ensemble.pipeline.stages.premarket import PremarketStage
from trading_ensemble.pipeline.stages.elimination import EliminationStage
from trading_ensemble.pipeline.stages.signals import SignalsStage
from trading_ensemble.pipeline.stages.risk import RiskStage
from trading_ensemble.pipeline.stages.execution import ExecutionStage


def main():

    stages = [
        PremarketStage(),
        EliminationStage(),
        SignalsStage(),
        RiskStage(),
        ExecutionStage(),
    ]

    engine = PipelineEngine(stages)

    engine.run()


if __name__ == "__main__":
    main()