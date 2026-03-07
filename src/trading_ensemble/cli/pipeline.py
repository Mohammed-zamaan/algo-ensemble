from trading_ensemble.config.settings import Settings
from trading_ensemble.pipeline.engine import PipelineEngine
from trading_ensemble.pipeline.stages.premarket import PremarketStage
from trading_ensemble.pipeline.stages.elimination import EliminationStage
from trading_ensemble.pipeline.stages.signals import SignalsStage
from trading_ensemble.pipeline.stages.risk import RiskStage
from trading_ensemble.pipeline.stages.execution import ExecutionStage
from trading_ensemble.state.store import StateStore


def main():
    settings = Settings.from_env()
    settings.validate_for_runtime()

    store = StateStore(settings.state_db_path)
    store.initialize()

    run_id = store.create_run(
        mode=settings.trade_mode,
        paper_trade=settings.paper_trade,
        status="STARTED",
    )

    context = {
        "settings": settings,
        "store": store,
        "run_id": run_id,
    }

    stages = [
        PremarketStage(),
        EliminationStage(),
        SignalsStage(),
        RiskStage(),
        ExecutionStage(),
    ]

    engine = PipelineEngine(stages)

    try:
        engine.run(context)
        store.finish_run(run_id, status="COMPLETED")
    except Exception as exc:
        store.finish_run(run_id, status="FAILED", notes=str(exc))
        raise


if __name__ == "__main__":
    main()