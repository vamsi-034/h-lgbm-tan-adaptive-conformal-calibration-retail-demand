import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = "sample_forecast.csv"
    if not os.path.exists(csv_file):
        # Fallback path if run from root
        alt_paths = [
            r"C:\Users\asus\.gemini\antigravity-ide\scratch\h_lgbm_tan_v4\sample_forecast.csv",
            r"C:\Users\asus\OneDrive\Desktop\Demand forcast\New folder\sample_forecast.csv"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                csv_file = p
                break

    df = pd.read_csv(csv_file)
    series_id = "16014"
    s = df[df["series_id"].astype(str) == series_id].head(60).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5.37), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.35, color="#d0d0d0", zorder=0)

    # Clean axes spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # 1. 80% Prediction Interval (Hybrid + ACI)
    interval_bars = ax.bar(
        s.index, 
        s["q90_aci"] - s["q10_aci"], 
        bottom=s["q10_aci"],
        width=0.6, 
        color="#F3B864", 
        alpha=0.95, 
        zorder=1
    )

    # 2. Actual Demand Bars
    actual_bars = ax.bar(
        s.index, 
        s["sales"], 
        width=0.6, 
        color="#9AA1B0", 
        zorder=2
    )

    # 3. Point Forecast Line with markers
    point_line = ax.plot(
        s.index, 
        s["point_hybrid"], 
        color="#D1451B", 
        linestyle="--", 
        linewidth=1.8,
        marker="o", 
        markersize=3.8, 
        zorder=4
    )

    # Limits and Ticks - fully preserves the 4,000 unit spike without clipping
    ax.set_xlim(-1, 60)
    ax.set_ylim(0, 4200)

    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks(range(0, 4001, 500))

    ax.set_xlabel("Time (days)", fontsize=12, labelpad=8)
    ax.set_ylabel("Sales volume", fontsize=12, labelpad=8)
    ax.set_title("H-LGBM-TAN forecast on lumpy demand (UCI item 16014, final rolling origin)",
                 fontsize=13, fontweight="bold", pad=12)

    # Legend
    handles = [interval_bars, point_line[0], actual_bars]
    labels = [
        "80% prediction interval (Hybrid + ACI)",
        "H-LGBM-TAN point forecast",
        "Actual demand"
    ]
    ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.95, edgecolor="#D6D6D6", fontsize=10.5)

    # Annotations with callout arrows
    ax.annotate(
        "Actual 500; forecast 450;\ninterval [2.9, 105.4]",
        xy=(6, 520), 
        xytext=(9.2, 1250),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
        fontsize=10.5, 
        color="#222222"
    )

    ax.annotate(
        "Actual 4,000; forecast 1,856;\ninterval [0.9, 70.1]",
        xy=(47.8, 3980), 
        xytext=(34.0, 2880),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
        fontsize=10.5, 
        color="#222222"
    )

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_files = [
        "figures/fig7_timeseries_forecast.png",
        "figures/fig7_timeseries_forecast.pdf",
        "figures/fig4_timeseries_forecast.png",
        "figures/fig4_timeseries_forecast.pdf",
        "figures/sample_forecast_plot.png",
        "figures/sample_forecast_plot.pdf"
    ]
    for out in out_files:
        if out.endswith(".png"):
            plt.savefig(out, dpi=300)
        else:
            plt.savefig(out)
        print(f"Saved: {out} ({os.path.getsize(out)} bytes)")

if __name__ == "__main__":
    main()
