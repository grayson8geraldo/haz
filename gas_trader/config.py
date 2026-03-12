"""Configuration loader — reads settings.yaml + .env and provides typed access."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"


def _load_yaml() -> dict:
    with open(CONFIG_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class InstrumentConfig:
    symbol: str
    exchange: str
    sec_type: str
    tick_size: float
    tick_value: float
    contract_size: int
    point_value: float
    margin_initial: float
    margin_maintenance: float
    contract_month: str = "auto"


@dataclass
class RiskConfig:
    risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_open_positions: int
    max_trades_per_day: int
    trailing_stop_enabled: bool
    trailing_activation_pct: float
    trailing_trail_pct: float
    eia_kill_switch: bool
    eia_pause_before: int
    eia_pause_after: int


@dataclass
class SignalWeights:
    technical: float
    fundamental: float
    ml: float
    seasonality: float


@dataclass
class Settings:
    initial_capital: float
    target_capital: float
    mode: str  # paper | live
    instrument: InstrumentConfig
    micro_instrument: InstrumentConfig
    risk: RiskConfig
    signal_weights: SignalWeights
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(CONFIG_DIR / ".env")
        raw = _load_yaml()

        acct = raw["account"]
        inst = raw["instrument"]
        micro = inst["micro"]
        risk_cfg = raw["risk"]
        ps = risk_cfg["position_sizing"]
        dl = risk_cfg["daily_limits"]
        ts = risk_cfg["trailing_stop"]
        eia_ks = risk_cfg["eia_kill_switch"]
        sig = raw["signals"]

        return cls(
            initial_capital=acct["initial_capital"],
            target_capital=acct["target_capital"],
            mode=acct["mode"],
            instrument=InstrumentConfig(
                symbol=inst["symbol"],
                exchange=inst["exchange"],
                sec_type=inst["sec_type"],
                tick_size=inst["tick_size"],
                tick_value=inst["tick_value"],
                contract_size=inst["contract_size"],
                point_value=inst["point_value"],
                margin_initial=inst["margin_initial"],
                margin_maintenance=inst["margin_maintenance"],
                contract_month=inst.get("contract_month", "auto"),
            ),
            micro_instrument=InstrumentConfig(
                symbol=micro["symbol"],
                exchange=micro["exchange"],
                sec_type=micro["sec_type"],
                tick_size=micro["tick_size"],
                tick_value=micro["tick_value"],
                contract_size=micro["contract_size"],
                point_value=micro["point_value"],
                margin_initial=micro["margin_initial"],
                margin_maintenance=micro["margin_maintenance"],
            ),
            risk=RiskConfig(
                risk_per_trade_pct=ps["risk_per_trade_pct"],
                max_risk_per_trade_pct=ps["max_risk_per_trade_pct"],
                atr_multiplier_sl=ps["atr_multiplier_sl"],
                atr_multiplier_tp=ps["atr_multiplier_tp"],
                max_daily_loss_pct=dl["max_daily_loss_pct"],
                max_drawdown_pct=dl["max_drawdown_pct"],
                max_open_positions=dl["max_open_positions"],
                max_trades_per_day=dl["max_trades_per_day"],
                trailing_stop_enabled=ts["enabled"],
                trailing_activation_pct=ts["activation_pct"],
                trailing_trail_pct=ts["trail_pct"],
                eia_kill_switch=eia_ks["enabled"],
                eia_pause_before=eia_ks["pause_minutes_before"],
                eia_pause_after=eia_ks["pause_minutes_after"],
            ),
            signal_weights=SignalWeights(
                technical=sig["technical"]["weight"],
                fundamental=sig["fundamental"]["weight"],
                ml=sig["ml"]["weight"],
                seasonality=sig["seasonality"]["weight"],
            ),
            raw=raw,
        )

    def select_instrument(self, capital: float) -> InstrumentConfig:
        """Выбирает NG или QG (micro) в зависимости от размера капитала."""
        if capital < self.instrument.margin_initial * 1.5:
            return self.micro_instrument
        return self.instrument
