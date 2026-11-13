#!/usr/bin/env python3
"""Create a visual, general-audience PDF deck for the multi-dataset study."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np


OUT = Path("docs/presentation_unified_aerial_detector")
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "unified_aerial_detector_general_audience.pdf"

BG = "#07111F"
PANEL = "#101F33"
PANEL2 = "#152942"
WHITE = "#F5F8FC"
MUTED = "#A9B8CC"
CYAN = "#28D7E6"
PURPLE = "#9B7BFF"
ORANGE = "#FFAE57"
GREEN = "#55E6A5"
RED = "#FF7085"


def canvas():
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 56.25); ax.axis("off")
    return fig, ax


def title(ax, kicker, headline, subtitle=None):
    ax.text(6, 51.2, kicker.upper(), color=CYAN, fontsize=10, fontweight="bold")
    ax.text(6, 46.3, headline, color=WHITE, fontsize=27, fontweight="bold", va="top")
    if subtitle:
        ax.text(6, 39.3, subtitle, color=MUTED, fontsize=12.5, va="top", linespacing=1.35)


def footer(ax, number):
    ax.text(94, 2.3, f"{number:02d}", color="#526783", fontsize=9, ha="right")


def rounded(ax, x, y, w, h, color=PANEL, radius=1.5, edge=None, lw=1):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={radius}",
                         facecolor=color, edgecolor=edge or color, linewidth=lw)
    ax.add_patch(box); return box


def sample_image(folder: str):
    paths = sorted(Path(folder).glob("*"))
    if not paths:
        return np.zeros((32, 32, 3))
    # Pick a visually legible example rather than an arbitrarily dark patch.
    candidates = paths[::max(1, len(paths)//60)][:60]
    images = [plt.imread(path) for path in candidates]
    scores = [float(image[..., :3].mean() + .35 * image[..., :3].std()) for image in images]
    return images[int(np.argmax(scores))]


def image_card(fig, ax, x, y, w, h, image, label, accent):
    rounded(ax, x, y, w, h, PANEL, edge=accent, lw=1.5)
    iax = fig.add_axes([x/100 + .008, y/56.25 + .035, w/100 - .016, h/56.25 - .075])
    iax.imshow(image[..., :3]); iax.axis("off")
    ax.text(x+1.2, y+h-1.6, label, color=WHITE, fontsize=10.5, fontweight="bold", va="top",
            bbox=dict(facecolor=accent, edgecolor="none", boxstyle="round,pad=.35"))


def save(fig, pdf, index):
    fig.savefig(OUT / f"slide_{index:02d}.png", dpi=150, facecolor=fig.get_facecolor())
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    fine = sample_image("/output/yolo_mtl_patches/fineair/rgbir/test/images")
    rare = sample_image("/output/yolo_mtl_patches/rareplanes/rgb/test/images")
    dota = sample_image("/output/yolo_mtl_patches/dota_aircraft/rgb/test/images")

    with PdfPages(PDF) as pdf:
        # 1 — Cover
        fig, ax = canvas()
        for cx, cy, r, c in [(82,44,13,CYAN),(91,14,17,PURPLE),(15,7,10,ORANGE)]:
            ax.add_patch(Circle((cx,cy), r, color=c, alpha=.08))
        ax.text(6, 49.5, "REMOTE SENSING • MULTI-DATASET LEARNING", color=CYAN,
                fontsize=10, fontweight="bold")
        ax.text(6, 39.5, "One model.\nThree aerial worlds.", color=WHITE,
                fontsize=39, fontweight="bold", va="top", linespacing=.95)
        ax.text(6, 22.3, "What we learned from unifying heterogeneous aircraft datasets",
                color=MUTED, fontsize=15)
        image_card(fig, ax, 62, 29, 14, 17, fine, "FineAir", CYAN)
        image_card(fig, ax, 77, 20, 14, 17, rare, "RarePlanes", PURPLE)
        image_card(fig, ax, 62, 9, 14, 17, dota, "DOTA", ORANGE)
        ax.text(6, 7.2, "SatellitePy research study", color=WHITE, fontsize=11, fontweight="bold")
        ax.text(6, 4.7, "Detection • fine-grained semantics • sparse supervision", color=MUTED, fontsize=9.5)
        save(fig, pdf, 1)

        # 2 — Problem
        fig, ax = canvas(); title(ax, "The challenge", "Aerial datasets speak different languages",
            "The imagery changes—and so do the labels. Valuable annotations stay trapped in separate models.")
        cards = [
            (6, 7, fine, "FineAir", "RGB + infrared", "aircraft type\nvery-fine type", CYAN),
            (37, 7, rare, "RarePlanes", "RGB", "role • engines\nwings • dimensions", PURPLE),
            (68, 7, dota, "DOTA", "RGB, different scale", "aircraft boxes\nobject masks", ORANGE),
        ]
        for x,y,img,name,sensor,labels,c in cards:
            image_card(fig, ax, x, y, 26, 27, img, name, c)
            fig.text((x+1.4)/100, (y+6.7)/56.25, sensor, color=c, fontsize=9.5,
                     fontweight="bold", bbox=dict(facecolor=BG, alpha=.88, edgecolor="none", pad=3))
            fig.text((x+1.4)/100, (y+1.2)/56.25, labels.replace("\n", "  •  "), color=WHITE,
                     fontsize=9.2, va="bottom",
                     bbox=dict(facecolor=BG, alpha=.88, edgecolor="none", pad=3))
        footer(ax, 2); save(fig, pdf, 2)

        # 3 — Solution diagram
        fig, ax = canvas(); title(ax, "Our approach", "Share visual understanding—route only known labels",
            "Missing annotations are ignored, never treated as background. Semantic tasks cannot destabilize detection.")
        for y,c,label in [(28,CYAN,"FineAir"),(20,PURPLE,"RarePlanes"),(12,ORANGE,"DOTA")]:
            rounded(ax, 6, y, 13, 5, PANEL2, edge=c, lw=1.5)
            ax.text(12.5,y+2.5,label,color=WHITE,fontsize=10.5,fontweight="bold",ha="center",va="center")
            ax.add_patch(FancyArrowPatch((19.5,y+2.5),(30,y+2.5),arrowstyle="-|>",mutation_scale=14,color=c,lw=2))
        rounded(ax, 31, 13, 19, 20, "#12304A", edge=CYAN, lw=1.6)
        ax.text(40.5,27.5,"SHARED",color=CYAN,fontsize=10,fontweight="bold",ha="center")
        ax.text(40.5,23.1,"YOLO11 visual\nbackbone + pyramid",color=WHITE,fontsize=15,fontweight="bold",ha="center",va="center")
        ax.text(40.5,16.3,"one visual representation",color=MUTED,fontsize=9.5,ha="center")
        ax.add_patch(FancyArrowPatch((50.5,23),(60,23),arrowstyle="-|>",mutation_scale=15,color=WHITE,lw=2))
        rounded(ax, 61, 25.5, 31, 7, PANEL2, edge=GREEN, lw=1.5)
        ax.text(63,29.6,"DETECTION",color=GREEN,fontsize=9.5,fontweight="bold")
        ax.text(78,29.2,"shared aircraft localization",color=WHITE,fontsize=11,ha="center")
        rounded(ax, 61, 15.5, 31, 7, PANEL2, edge=PURPLE, lw=1.5)
        ax.text(63,19.6,"SEMANTICS",color=PURPLE,fontsize=9.5,fontweight="bold")
        ax.text(78,19.2,"dataset-aware sparse heads",color=WHITE,fontsize=11,ha="center")
        rounded(ax, 61, 5.5, 31, 7, PANEL2, edge=ORANGE, lw=1.5)
        ax.text(63,9.6,"SAFETY",color=ORANGE,fontsize=9.5,fontweight="bold")
        ax.text(78,9.2,"detached semantic gradients",color=WHITE,fontsize=11,ha="center")
        footer(ax, 3); save(fig, pdf, 3)

        # 4 — Full-data result
        fig, ax = canvas(); title(ax, "Core result", "One shared detector stays near three specialists",
            "Three-seed test mAP50–95. Differences are small compared with seed-to-seed variation.")
        datasets = ["FineAir", "RarePlanes", "DOTA"]
        indep = np.array([.9324,.7876,.8101]); shared=np.array([.9348,.7909,.8090])
        err_i=np.array([.0059,.0052,.0068]); err_s=np.array([.0017,.0002,.0095])
        bax=fig.add_axes([.08,.15,.61,.47],facecolor=BG); x=np.arange(3); bw=.34
        bax.bar(x-bw/2,indep,bw,color="#4C607B",label="Independent specialists",yerr=err_i,capsize=3)
        bax.bar(x+bw/2,shared,bw,color=CYAN,label="One shared model",yerr=err_s,capsize=3)
        bax.set_ylim(.72,.97); bax.set_xticks(x,datasets); bax.tick_params(colors=WHITE,labelsize=10)
        bax.spines[['top','right','left']].set_visible(False); bax.spines['bottom'].set_color('#51647D')
        bax.grid(axis='y',color='white',alpha=.08); bax.set_axisbelow(True)
        bax.legend(frameon=False,labelcolor=WHITE,loc='upper right')
        for i,v in enumerate(shared): bax.text(i+bw/2,v+.009,f"{v:.3f}",ha='center',color=WHITE,fontsize=9,fontweight='bold')
        rounded(ax, 73, 11, 21, 22, PANEL2, edge=GREEN, lw=1.7)
        ax.text(83.5,28.8,"≈ 63%",color=GREEN,fontsize=30,fontweight="bold",ha="center")
        ax.text(83.5,24.5,"less checkpoint storage",color=WHITE,fontsize=11,fontweight="bold",ha="center")
        ax.text(83.5,18.8,"136 MB shared\nvs 369 MB for 3 specialists",color=MUTED,fontsize=10,ha="center",linespacing=1.35)
        ax.text(83.5,13.8,"One interface • one deployment",color=CYAN,fontsize=9.5,ha="center")
        footer(ax, 4); save(fig, pdf, 4)

        # 5 — Low data
        fig, ax = canvas(); title(ax, "A promising signal", "With only 25% of labels, sharing helps FineAir",
            "Seed-0 test mAP50–95 on unchanged test sets. The overall gain is modest—but data sharing can help the smallest domain.")
        labels=["FineAir","RarePlanes","DOTA","Macro mean"]
        i=np.array([.9021,.7261,.7969,.8083]); s=np.array([.9180,.7206,.7980,.8122])
        bax=fig.add_axes([.08,.14,.67,.48],facecolor=BG); x=np.arange(4); bw=.34
        bax.bar(x-bw/2,i,bw,color="#4C607B",label="Independent 25%")
        bax.bar(x+bw/2,s,bw,color=PURPLE,label="Shared 25%")
        bax.set_ylim(.68,.95); bax.set_xticks(x,labels); bax.tick_params(colors=WHITE,labelsize=10)
        bax.spines[['top','right','left']].set_visible(False); bax.spines['bottom'].set_color('#51647D')
        bax.grid(axis='y',color='white',alpha=.08); bax.set_axisbelow(True)
        bax.legend(frameon=False,labelcolor=WHITE,loc='upper right')
        rounded(ax, 79, 12, 15, 21, PANEL2, edge=CYAN, lw=1.6)
        ax.text(86.5,27.8,"+1.59",color=CYAN,fontsize=29,fontweight="bold",ha="center")
        ax.text(86.5,23.9,"AP points",color=WHITE,fontsize=11,fontweight="bold",ha="center")
        ax.text(86.5,19.5,"FineAir gain",color=MUTED,fontsize=10,ha="center")
        ax.text(86.5,15.2,"+0.39 macro",color=GREEN,fontsize=10,fontweight="bold",ha="center")
        footer(ax, 5); save(fig, pdf, 5)

        # 6 — Semantics
        fig, ax = canvas(); title(ax, "Beyond a bounding box", "The same model answers richer questions",
            "Oracle-box semantic evaluation: labels are sparse, but the model learns whichever annotations each dataset provides.")
        metrics=[("FineAir\nfine type",48.0,19.4,CYAN),("FineAir\nvery-fine",53.5,23.9,PURPLE),
                 ("RarePlanes\nrole",89.1,55.2,ORANGE),("RarePlanes\nengine type",90.6,57.9,GREEN)]
        for idx,(name,val,base,c) in enumerate(metrics):
            x=6+23.5*idx; rounded(ax,x,10,20,25,PANEL2,edge=c,lw=1.4)
            ax.text(x+10,30.2,name,color=WHITE,fontsize=11,fontweight="bold",ha="center",va="center")
            ax.text(x+10,21.9,f"{val:.1f}%",color=c,fontsize=25,fontweight="bold",ha="center")
            ax.text(x+10,17.7,"model accuracy",color=MUTED,fontsize=8.8,ha="center")
            ax.plot([x+3,x+17],[14.6,14.6],color="#40536B",lw=5,solid_capstyle='round')
            ax.plot([x+3,x+3+14*base/100],[14.6,14.6],color=RED,lw=5,solid_capstyle='round')
            ax.text(x+10,11.5,f"majority baseline {base:.1f}%",color=MUTED,fontsize=8,ha="center")
        footer(ax, 6); save(fig, pdf, 6)

        # 7 — Close
        fig, ax = canvas(); title(ax, "What is genuinely positive", "Unification works—even when free accuracy does not",
            "The contribution is a practical and evidence-based framework for heterogeneous aerial learning.")
        items=[
            ("01","Consolidation", "One model approaches three specialist detectors.", CYAN),
            ("02","Annotation reuse", "Fine classes, attributes and masks coexist without fabricated labels.", PURPLE),
            ("03","Operational simplicity", "One checkpoint, one interface, fewer models to maintain.", GREEN),
            ("04","Scientific clarity", "We map where sharing helps—and where domain mismatch still wins.", ORANGE),
        ]
        for j,(num,head,body,c) in enumerate(items):
            x=6+(j%2)*46; y=25-(j//2)*13
            rounded(ax,x,y,41,10,PANEL2,edge=c,lw=1.3)
            ax.text(x+3,y+6.4,num,color=c,fontsize=15,fontweight="bold")
            ax.text(x+10,y+6.6,head,color=WHITE,fontsize=12,fontweight="bold")
            ax.text(x+10,y+3.7,"\n".join(textwrap.wrap(body,48)),color=MUTED,fontsize=9.2,va="top")
        ax.text(6,5.5,"TAKEAWAY",color=CYAN,fontsize=9.5,fontweight="bold")
        ax.text(19,5.5,"Shared learning is most valuable as a unification strategy—not a guaranteed accuracy boost.",
                color=WHITE,fontsize=12,fontweight="bold")
        footer(ax, 7); save(fig, pdf, 7)

    print(PDF)


if __name__ == "__main__":
    main()
