
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phaseflow-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def configure_style(font: Path | None = None) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    if font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        fontManager.addfont(str(font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font)).get_name()]


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"metrics input does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"metrics input contains no rows: {path}")
    return frame


def choose_column(frame: pd.DataFrame, requested: str | None, candidates: Iterable[str], kind: str) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{kind} column is missing: {requested}")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"could not infer {kind} column; pass the corresponding command-line option")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 300, "facecolor": "white"}),
        (".pdf", {"facecolor": "white"}),
        (".svg", {"facecolor": "none", "transparent": True}),
    ):
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight", pad_inches=0.16, **kwargs)
    plt.close(fig)
