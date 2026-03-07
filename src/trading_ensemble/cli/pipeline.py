from trading_ensemble.config.settings import Settings
from trading_ensemble.pipeline.engine import PipelineEngine

from trading_ensemble.pipeline.stages.premarket import PremarketStage
from trading_ensemble.pipeline.stages.elimination import EliminationStage
from trading_ensemble.pipeline.stages.signals import SignalsStage
from trading_ensemble.pipeline.stages.risk import RiskStage
from trading_ensemble.pipeline.stages.execution import ExecutionStage


def main():

    # ---------------------------------------------------------
    # Load system configuration
    # ---------------------------------------------------------
    settings = Settings.from_env()
    settings.validate_for_runtime()

    context = {
        "settings": settings
    }

    # ---------------------------------------------------------
    # Define pipeline stages
    # ---------------------------------------------------------
    stages = [
        PremarketStage(),
        EliminationStage(),
        SignalsStage(),
        RiskStage(),
        ExecutionStage(),
    ]

    engine = PipelineEngine(stages)

    engine.run(context)


if __name__ == "__main__":
    main()