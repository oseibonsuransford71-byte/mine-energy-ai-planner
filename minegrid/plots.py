"""Figures for the README. Static PNGs, light surface, validated categorical palette.

Colour rules: categorical hues are assigned in fixed order (blue, orange, aqua), baselines are
neutral greys, diverging data uses blue/red around a grey midpoint, magnitude uses one blue ramp.
Every multi-series chart has a legend; text is always in ink colours, never series colours.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY_DARK, GREY_LIGHT = "#898781", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DPI = 160

MODEL_COLOR = {
    "Climatology": GREY_LIGHT,
    "Smart persistence": GREY_DARK,
    "Same hour yesterday": GREY_DARK,
    "Same hour last week": GREY_LIGHT,
    "Ridge": ORANGE,
    "Gradient boosting": BLUE,
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASE,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_dispatch(detail_dry: pd.DataFrame, detail_wet: pd.DataFrame, path, design_label: str):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharey=True)
    for ax, d, title in zip(axes, (detail_dry, detail_wet), ("Dry-season week (January)", "Wet-season week (July)")):
        x = d.index
        ax.stackplot(
            x,
            d["pv_used"],
            d["batt_discharge"],
            d["diesel"],
            colors=[BLUE, AQUA, ORANGE],
            labels=["Solar PV", "Battery discharge", "Diesel"],
            linewidth=0.5,
            edgecolors=SURFACE,
        )
        ax.plot(x, d["load"], color=INK, linewidth=1.1, label="Mine load")
        ax.set_title(title)
        ax.set_ylabel("MW")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 22)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.suptitle(f"Hourly dispatch, {design_label}", x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, path)


def fig_forecast_skill(metrics: pd.DataFrame, path):
    _style()
    horizons = sorted(metrics["horizon_h"].unique())
    models = ["Climatology", "Smart persistence", "Ridge", "Gradient boosting"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.2
    for i, m in enumerate(models):
        vals = [metrics[(metrics.model == m) & (metrics.horizon_h == h)]["nRMSE_%"].iloc[0] for h in horizons]
        ax.bar(np.arange(len(horizons)) + (i - 1.5) * w, vals, width=w - 0.02, color=MODEL_COLOR[m], label=m)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h} h ahead" for h in horizons])
    ax.set_ylabel("nRMSE (% of mean daytime GHI), lower is better")
    ax.set_title("Solar irradiance forecast error on the 2023 test year")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_ylim(0, 56)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_solar_example(preds: dict, start: str, days: int, path):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    for ax, h in zip(axes, (6, 24)):
        obs = preds[("solar_obs", "Observed", h)].loc[start:end]
        fc = preds[("solar", "Gradient boosting", h)].reindex(obs.index)
        lo = preds[("solar_lo", "Gradient boosting", h)].reindex(obs.index)
        hi = preds[("solar_hi", "Gradient boosting", h)].reindex(obs.index)
        full = pd.date_range(start, end, freq="h")
        obs, fc, lo, hi = (s.reindex(full).fillna(0.0) for s in (obs, fc, lo, hi))
        ax.fill_between(full, lo, hi, color=BLUE, alpha=0.18, linewidth=0, label="80% prediction interval")
        ax.plot(full, fc, color=BLUE, label="Forecast (gradient boosting)")
        ax.plot(full, obs, color=INK, linewidth=1.0, label="Observed")
        ax.set_ylabel("GHI (W/m²)")
        ax.set_title(f"{h} h ahead")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    axes[1].xaxis.set_major_locator(matplotlib.dates.DayLocator())
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.tight_layout()
    _save(fig, path)


def fig_load_example(preds: dict, start: str, path):
    _style()
    end = pd.Timestamp(start) + pd.Timedelta(days=7)
    obs = preds[("load_obs", "Observed", 24)].loc[start:end]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(obs.index, preds[("load", "Same hour yesterday", 24)].reindex(obs.index), color=GREY_DARK, linewidth=1.0, label="Same hour yesterday")
    ax.plot(obs.index, preds[("load", "Gradient boosting", 24)].reindex(obs.index), color=BLUE, label="Day-ahead forecast (gradient boosting)")
    ax.plot(obs.index, obs, color=INK, linewidth=1.0, label="Observed")
    ax.set_ylabel("Mine load (MW)")
    ax.set_title("Day-ahead load forecast, one week of the 2023 test year")
    ax.legend(ncol=3, loc="lower center", frameon=False)
    ax.set_ylim(11, 19)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    _save(fig, path)


def fig_design_space(priced: pd.DataFrame, best: pd.Series, path):
    _style()
    sav = priced.pivot(index="bess_mwh", columns="pv_mwp", values="saving_usd_yr") / 1e6
    ren = priced.pivot(index="bess_mwh", columns="pv_mwp", values="ren_fraction") * 100
    div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#184f95"])
    seq = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [
        (axes[0], sav, div, TwoSlopeNorm(vmin=min(sav.values.min(), -1), vcenter=0, vmax=sav.values.max()), "Annual saving vs diesel-only (M USD)"),
        (axes[1], ren, seq, None, "Renewable share of mine demand (%)"),
    ]
    for ax, data, cmap, norm, title in panels:
        im = ax.imshow(data.values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([int(c) for c in data.columns])
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([int(i) for i in data.index])
        ax.set_xlabel("PV capacity (MWp)")
        ax.set_ylabel("Battery energy (MWh)")
        ax.set_title(title)
        ax.grid(False)
        bx = list(data.columns).index(best["pv_mwp"])
        by = list(data.index).index(best["bess_mwh"])
        ax.scatter([bx], [by], s=140, facecolors="none", edgecolors=INK, linewidths=1.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].annotate(
        f"cost-optimal:\n{int(best['pv_mwp'])} MWp + {int(best['bess_mwh'])} MWh",
        xy=(list(sav.columns).index(best["pv_mwp"]), list(sav.index).index(best["bess_mwh"])),
        xytext=(2.0, 11.0),
        color=INK,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
    )
    fig.tight_layout()
    _save(fig, path)


def fig_frontier(frontier: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    f = frontier.drop_duplicates(subset=["pv_mwp", "bess_mwh"]).reset_index(drop=True)
    ax.plot(f["ren_fraction"] * 100, f["saving_musd"], color=BLUE, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for _, r in f.iloc[1:].iloc[::2].iterrows():
        ax.annotate(
            f"{int(r.pv_mwp)} MWp / {int(r.bess_mwh)} MWh",
            (r.ren_fraction * 100, r.saving_musd),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=8,
            color=INK2,
        )
    ax.set_xlabel("Renewable share of mine demand (%)")
    ax.set_ylabel("Annual saving vs diesel-only (M USD)")
    ax.set_title("Diminishing returns: the cheapest design for each renewable-share cap")
    ax.set_ylim(0, None)
    _save(fig, path)


def fig_sensitivity(sens: pd.DataFrame, path):
    _style()
    s = sens[sens.capex_mult == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {0.1: BLUE, 0.3: ORANGE, 0.5: AQUA}
    for share, g in s.groupby("min_diesel_share"):
        g = g.sort_values("diesel_price")
        label = f"{int(share * 100)}% minimum diesel share"
        for ax, col, scale in ((axes[0], "saving_musd", 1.0), (axes[1], "ren_fraction", 100.0)):
            ax.plot(g["diesel_price"], g[col] * scale, color=colors[share], label=label)
            solid = g[~g["at_grid_edge"]]
            edge = g[g["at_grid_edge"]]
            ax.plot(solid["diesel_price"], solid[col] * scale, "o", color=colors[share], markersize=5)
            # hollow marker: optimum is on the edge of the search grid, so the value is a lower bound
            ax.plot(edge["diesel_price"], edge[col] * scale, "o", markerfacecolor=SURFACE, markeredgecolor=colors[share], markersize=6, markeredgewidth=1.5)
    axes[0].set_ylabel("Annual saving at optimum (M USD)")
    axes[1].set_ylabel("Renewable share at optimum (%)")
    for ax in axes:
        ax.set_xlabel("Delivered diesel price (USD/L)")
    axes[0].set_title("Savings scale with fuel price")
    axes[1].set_title("Reserve rule caps renewable share")
    axes[0].legend(loc="upper left")
    fig.text(0.01, -0.02, "Hollow markers: optimum sits on the edge of the search grid (100 MWp / 240 MWh), so values are lower bounds.", fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, path)


def fig_forecast_value(table: pd.DataFrame, path):
    _style()
    t = table.iloc[1:].copy()
    colors = {
        "ML forecast (gradient boosting)": BLUE,
        "ML forecast (ridge)": ORANGE,
        "Persistence (same hour yesterday)": GREY_DARK,
        "No forecast (assume zero solar)": GREY_LIGHT,
    }
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["excess_vs_oracle_kusd"] / 1e3, color=[colors[s] for s in t["scenario"]], height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(t["scenario"])
    for yi, v in zip(y, t["excess_vs_oracle_kusd"] / 1e3):
        ax.text(v + 0.08, yi, f"${v:.2f}M", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Extra fuel cost per year vs perfect-foresight commitment (M USD)")
    ax.set_title("What forecast quality is worth in genset commitment")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t["excess_vs_oracle_kusd"].max() / 1e3 * 1.18)
    _save(fig, path)
