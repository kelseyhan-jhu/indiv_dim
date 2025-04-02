
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.idm._definition import BinnedSeries
import numpy as np

DEFAULT_MATPLOTLIBRC = {
    "font.family": ["sans-serif", "serif"],
    "font.serif": ["NewComputerModernMath"],
    "font.sans-serif": ["NewComputerModernSans10"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.figsize": (3, 3),
    "figure.autolayout": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "savefig.format": "svg",
    "savefig.transparent": True,
    "figure.facecolor": "None",
    "axes.facecolor": "None",
    #"axes.labelsize": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "figure.titlesize": "medium",
    "figure.labelsize": "medium",
    "legend.title_fontsize": "medium",
    "legend.fontsize": "medium",
    "legend.fancybox": False,
    "legend.frameon": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "patch.linewidth": 0,
}



def plot_spectrum(ax, data, idx, **kwargs):
    
    label = kwargs['label'] if 'label' in kwargs else None
    color = kwargs['color'] if 'color' in kwargs else None
    if type(data) is dict:
        ax.errorbar(
            data['ranks'] * np.power(10, 0.03 * idx),
            data['means'],
            data['stds'],
            marker='o',
            markersize=8,
            linestyle='',
            label=label,
            color=color,
        )
    elif type(BinnedSeries):
        ax.errorbar(
            data.ranks * np.power(10, 0.03 * idx),
            data.means,
            data.stds,
            marker='o',
            markersize=8,
            linestyle='',
            label=label,
            color=color,
        )
