"""Shared matplotlib styling and color palette for all figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Color palette — colorblind-friendly (Okabe-Ito)
COLORS = {
    "naive": "#E69F00",            # orange
    "vts_only": "#56B4E9",         # sky blue
    "vts_dedup": "#009E73",        # teal
    "dce": "#D55E00",              # vermillion
    # Ablation methods
    "dedup_only": "#CC79A7",       # reddish purple
    "prompt_evo_only": "#0072B2",  # bluish green
    "prompt_evo_dedup": "#F0E442", # yellow
    # Token-level baselines
    "temp_1.2_dedup": "#999999",   # gray
    "nucleus_0.9_dedup": "#661100",# dark red
}

LABELS = {
    "naive": "Naive",
    "vts_only": "VTS Only",
    "vts_dedup": "VTS + Dedup",
    "dce": "DCE (Full)",
    "dedup_only": "Dedup Only",
    "prompt_evo_only": "Prompt Evo Only",
    "prompt_evo_dedup": "Prompt Evo + Dedup",
    "temp_1.2_dedup": "Temp 1.2 + Dedup",
    "nucleus_0.9_dedup": "Nucleus 0.9 + Dedup",
}

FIGURE_DIR = "paper/figures"


def setup_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
    })


def save_figure(fig: plt.Figure, name: str):
    """Save figure as PDF and generate a pgfplots .tex file."""
    from pathlib import Path
    out = Path(FIGURE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    print(f"Saved {out / name}.pdf")

    # Generate pgfplots .tex from axes data
    tex = _fig_to_pgfplots(fig)
    if tex:
        (out / f"{name}.tex").write_text(tex)
        print(f"Saved {out / name}.tex")
    plt.close(fig)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to pgfplots rgb,255:red,R;green,G;blue,B format."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgb,255:red,{r};green,{g};blue,{b}"


def _fig_to_pgfplots(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a standalone pgfplots .tex snippet."""
    n_axes = len(fig.axes)
    lines = [
        r"% Auto-generated pgfplots figure",
    ]

    # Use groupplot for multi-panel figures
    if n_axes > 1:
        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{groupplot}[")
        lines.append(f"  group style={{group size={n_axes} by 1, horizontal sep=2cm}},")
        lines.append("  width=0.45\\textwidth,")
        lines.append("  height=6cm,")
        lines.append("]")
    else:
        lines.append(r"\begin{tikzpicture}")

    color_idx = 0
    color_defs = []

    for ax in fig.axes:
        xlabel = ax.get_xlabel() or ""
        ylabel = ax.get_ylabel() or ""
        title = ax.get_title() or ""

        if n_axes > 1:
            lines.append(r"\nextgroupplot[")
        else:
            lines.append(r"\begin{axis}[")
            lines.append("  width=0.85\\textwidth,")
            lines.append("  height=7cm,")

        lines.append(f"  xlabel={{{xlabel}}},")
        lines.append(f"  ylabel={{{ylabel}}},")
        if title:
            lines.append(f"  title={{{title}}},")
        lines.append("  legend pos=north east,")
        lines.append("  grid=major,")
        lines.append("]")

        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            label = line.get_label()
            hex_color = mpl.colors.to_hex(line.get_color())
            if len(xdata) == 0 or label.startswith("_"):
                continue
            # Define a named color to avoid # in the tex
            cname = f"plotcolor{color_idx}"
            color_defs.append(f"\\definecolor{{{cname}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}")
            color_idx += 1
            lines.append(f"\\addplot[color={cname}, thick] coordinates {{")
            step = max(1, len(xdata) // 200)
            for x, y in zip(xdata[::step], ydata[::step]):
                lines.append(f"  ({x}, {y:.6f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{label}}}")

        if n_axes == 1:
            lines.append(r"\end{axis}")

    if n_axes > 1:
        lines.append(r"\end{groupplot}")

    lines.append(r"\end{tikzpicture}")

    # Prepend color definitions
    return "\n".join(color_defs + [""] + lines)
