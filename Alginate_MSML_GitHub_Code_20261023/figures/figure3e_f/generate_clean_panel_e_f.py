from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import argparse

outdir=Path(__file__).resolve().parent

def make_panel_e(outdir):
    cols=[0,20,40,50,60,80,100]
    rows=[f"Fold {i}" for i in range(1,8)]
    fig, ax = plt.subplots(figsize=(10.8,6.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-1.8, len(cols)+0.55)
    ax.set_ylim(-1.0, len(rows)+1.2)
    ax.invert_yaxis()
    ax.axis("off")
    train_face="#dbe5ea"; train_edge="#1b8a95"; test_face="#f4e3de"; test_edge="#cf5a43"; text_col="#1f3850"
    w,h=0.82,0.86; xoff=0.3
    for j,c in enumerate(cols):
        ax.text(j+xoff+w/2, 0.72, f"{c}%", ha="center", va="center", fontsize=14, fontweight="bold", color=text_col)
    for i,row in enumerate(rows, start=1):
        y=i
        ax.text(-0.68, y+h/2, row, ha="right", va="center", fontsize=14.5, color=text_col)
        for j,c in enumerate(cols):
            is_test = (i-1)==j
            face = test_face if is_test else train_face
            edge = test_edge if is_test else train_edge
            label = "TEST" if is_test else "TRAIN"
            weight = "bold" if is_test else "normal"
            color = test_edge if is_test else text_col
            rect = Rectangle((j+xoff, y), w, h, facecolor=face, edgecolor=edge, linewidth=2.0)
            ax.add_patch(rect)
            ax.text(j+xoff+w/2, y+h/2, label, ha="center", va="center", fontsize=13.5, color=color, fontweight=weight)
    handles=[Patch(facecolor=train_face, edgecolor=train_edge, label="Six training compositions"), Patch(facecolor=test_face, edgecolor=test_edge, label="One held-out composition")]
    leg=ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=14, handlelength=1.5, columnspacing=4)
    for text in leg.get_texts():
        text.set_color(text_col)
    fig.subplots_adjust(bottom=0.18, top=0.96)
    for ext in ["png","svg","pdf"]:
        path=outdir/f"Figure3e_clean_no_title.{ext}"
        fig.savefig(path, dpi=600 if ext=="png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def make_panel_f(outdir, data_file=None):
    if data_file is None:
        data_file=Path(__file__).resolve().parent / "source_data" / "V70_FIG3F_LOCKED_15_CONDITIONS.csv"
    df = pd.read_csv(data_file)
    piv = df.pivot(index="nce", columns="dp", values="observed_MAE_pp").sort_index().reindex(columns=[3,4,5,6,7])
    nce_labels=[f"{int(v)}%" for v in piv.index]
    dp_labels=[f"DP{int(v)}" for v in piv.columns]
    data=piv.values
    fig, ax = plt.subplots(figsize=(9.6,5.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="Greens", aspect="auto", vmin=0, vmax=10)
    ax.set_xticks(np.arange(len(dp_labels)))
    ax.set_xticklabels(dp_labels, fontsize=16)
    ax.set_yticks(np.arange(len(nce_labels)))
    ax.set_yticklabels(nce_labels, fontsize=16)
    ax.set_xlabel("Degree of polymerization", fontsize=18, labelpad=10)
    ax.set_ylabel("Normalized collision energy", fontsize=18, labelpad=12)
    ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=16, color="black")
    cbar=fig.colorbar(im, ax=ax, fraction=0.048, pad=0.06)
    cbar.set_label("MAE (M percentage points)", fontsize=18, labelpad=14)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    for ext in ["png","svg","pdf"]:
        path=outdir/f"Figure3f_clean_no_title.{ext}"
        fig.savefig(path, dpi=600 if ext=="png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Export manuscript Figure 3e/3f as PNG, SVG, and PDF from source data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "exports")
    parser.add_argument("--data-file", type=Path,
                        default=Path(__file__).resolve().parent / "source_data" / "V70_FIG3F_LOCKED_15_CONDITIONS.csv")
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    make_panel_e(args.out_dir)
    make_panel_f(args.out_dir,args.data_file)
