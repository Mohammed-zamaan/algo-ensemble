# backtest_chart.py — Equity curve + monthly PnL + distribution charts
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

def main():
    eq  = pd.read_csv("state/backtest_equity.csv",  parse_dates=["date"])
    tr  = pd.read_csv("state/backtest_trades.csv",  parse_dates=["date"])
    with open("state/backtest_metrics.json") as f:
        m = json.load(f)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Portfolio Equity Curve",
            "Monthly PnL (INR)",
            "Trade PnL Distribution",
            "Cumulative Return %"
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # 1. Equity curve
    fig.add_trace(go.Scatter(
        x=eq["date"], y=eq["equity"], mode="lines", name="Equity",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"
    ), row=1, col=1)

    # 2. Monthly PnL
    tr["month"] = tr["date"].dt.to_period("M").astype(str)
    monthly = tr.groupby("month")["pnl"].sum().reset_index()
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["pnl"], name="Monthly PnL",
        marker_color=["#00c853" if v >= 0 else "#ff1744" for v in monthly["pnl"]]
    ), row=1, col=2)

    # 3. Distribution
    wins   = tr[tr["pnl"] > 0]["pnl"]
    losses = tr[tr["pnl"] <= 0]["pnl"]
    fig.add_trace(go.Histogram(x=wins,   name="Wins",   marker_color="#00c853", opacity=0.75, nbinsx=25), row=2, col=1)
    fig.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#ff1744", opacity=0.75, nbinsx=25), row=2, col=1)

    # 4. Cumulative return
    eq["cum_ret"] = (eq["equity"] / eq["equity"].iloc[0] - 1) * 100
    fig.add_trace(go.Scatter(
        x=eq["date"], y=eq["cum_ret"], mode="lines", name="Cum Return",
        line=dict(color="#ffd700", width=2)
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=2, col=2)

    title = (f"algo-ensemble Backtest 2023-2026 | "
             f"Return: {m.get('Total Return %')}% | "
             f"Win: {m.get('Win Rate %')}% | "
             f"PF: {m.get('Profit Factor')} | "
             f"Sharpe: {m.get('Sharpe Ratio')} | "
             f"MaxDD: -{m.get('Max Drawdown %')}%")

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="white")),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#cccccc"),
        height=800, width=1400,
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
        barmode="overlay",
    )
    fig.update_xaxes(gridcolor="#1e1e1e", zeroline=False)
    fig.update_yaxes(gridcolor="#1e1e1e", zeroline=False)
    os.makedirs("state", exist_ok=True)
    fig.write_image("state/backtest_chart.png")
    print("Saved: state/backtest_chart.png")

if __name__ == "__main__":
    main()
    