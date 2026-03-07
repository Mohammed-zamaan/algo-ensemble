from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class CandidateSignal:
    symbol: str
    entry_price: float
    stop_loss: float
    conviction: float
    sector: str = "UNKNOWN"


@dataclass(frozen=True)
class AllocationDecision:
    symbol: str
    quantity: int
    allocation_amount: float
    stop_loss: float
    entry_price: float
    sector: str
    conviction: float


@dataclass(frozen=True)
class AllocatorConfig:
    max_gross_exposure_pct: float = 0.90
    cash_buffer_pct: float = 0.10
    max_single_position_pct: float = 0.12
    max_sector_exposure_pct: float = 0.25
    max_positions: int = 5


class AccountAllocator:
    def __init__(self, total_account_capital: float, config: AllocatorConfig | None = None) -> None:
        self.total_account_capital = total_account_capital
        self.config = config or AllocatorConfig()

    def allocate(self, candidates: Iterable[CandidateSignal]) -> List[AllocationDecision]:
        ranked = sorted(candidates, key=lambda x: x.conviction, reverse=True)
        ranked = ranked[: self.config.max_positions]

        deployable_capital = self.total_account_capital * (
            self.config.max_gross_exposure_pct - self.config.cash_buffer_pct
        )
        max_per_position = self.total_account_capital * self.config.max_single_position_pct

        decisions: List[AllocationDecision] = []
        remaining = deployable_capital

        for candidate in ranked:
            if candidate.entry_price <= 0:
                continue

            allocation_amount = min(max_per_position, remaining)
            quantity = int(allocation_amount // candidate.entry_price)

            if quantity <= 0:
                continue

            final_amount = quantity * candidate.entry_price

            decisions.append(
                AllocationDecision(
                    symbol=candidate.symbol,
                    quantity=quantity,
                    allocation_amount=final_amount,
                    stop_loss=candidate.stop_loss,
                    entry_price=candidate.entry_price,
                    sector=candidate.sector,
                    conviction=candidate.conviction,
                )
            )

            remaining -= final_amount
            if remaining <= 0:
                break

        return decisions