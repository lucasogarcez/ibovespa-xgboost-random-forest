"""Gera as 3 figuras conceituais (originais) do artigo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family": "serif", "font.size": 10})
G, O, Y, K, L = "#1b7f5c", "#c4511f", "#d9a400", "#333333", "#eef3f1"


def box(ax, x, y, w, h, text, fc=L, ec=K, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=K,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=K, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def tree(ax, cx, cy, color):
    pts = [(cx, cy + .16), (cx - .09, cy), (cx + .09, cy), (cx - .14, cy - .16), (cx - .04, cy - .16), (cx + .05, cy - .16), (cx + .14, cy - .16)]
    for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=K, lw=.8, zorder=1)
    for i, (px, py) in enumerate(pts):
        ax.add_patch(Circle((px, py), .022, fc=color if i > 2 else "white", ec=K, lw=.8, zorder=2))


# ---------------- Figura 1: Random Forest ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.1, 1.85, 1.5, 0.9, "Base de\ntreinamento", fc="#e8e8e8", bold=True)
ys = [3.35, 1.95, 0.55]
for i, yy in enumerate(ys):
    arrow(ax, (1.6, 2.3), (2.55, yy + .45))
    box(ax, 2.55, yy, 1.7, .9, f"Amostra bootstrap {i+1 if i<2 else 'B'}\n(com reposição)")
    arrow(ax, (4.25, yy + .45), (5.0, yy + .45))
    tree(ax, 5.45, yy + .45, G)
    ax.text(5.45, yy - .02, f"Árvore {i+1 if i<2 else 'B'}", ha="center", fontsize=8)
    arrow(ax, (5.95, yy + .45), (6.8, 2.3), rad=0.0)
    ax.text(6.35, yy + .6, ["Alta", "Não alta", "Alta"][i], fontsize=8, color=O, ha="center")
for k in range(3): ax.add_patch(Circle((3.4, 1.62 + k*0.12 - 0.0), .02, fc=K, ec=K))
box(ax, 6.8, 1.85, 1.5, .9, "Votação\nmajoritária", fc="#fbe9e1", bold=True)
arrow(ax, (8.3, 2.3), (8.85, 2.3))
box(ax, 8.85, 1.85, 1.05, .9, "Previsão\nfinal", fc="#fff3c4", bold=True)
ax.text(5, 4.45, "Atributos sorteados em cada divisão de nó (≈ √d)", ha="center", fontsize=8.5, style="italic")
fig.tight_layout(); fig.savefig("figuras/fig1_random_forest.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------- Figura 2: Boosting ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")
xs = [0.2, 3.6, 7.0]; W = 2.8
labels = ["Árvore 1\n$f_1$: modelo inicial", "Árvore 2\n$f_2$: corrige os\nerros de $f_1$", "Árvore M\n$f_M$: corrige os\nerros acumulados"]
for i, x in enumerate(xs):
    box(ax, x, 2.2, W, 1.3, "", fc=L)
    tree(ax, x + .55, 2.85, O)
    ax.text(x + 1.85, 2.85, labels[i], ha="center", va="center", fontsize=8.5)
    arrow(ax, (x + W / 2, 2.2), (x + W / 2, 1.35))
    if i < 2:
        arrow(ax, (x + W, 2.85), (xs[i + 1], 2.85))
        ax.text(x + W + .3, 3.65, "erros\nresiduais" if i == 0 else "erros\nresiduais", ha="center", va="center", fontsize=7.5, style="italic")
box(ax, 1.5, .35, 7.0, 1.0, "Previsão final:  $\\hat{y} = \\eta f_1(x) + \\eta f_2(x) + \\cdots + \\eta f_M(x)$", fc="#fff3c4", fs=10)
ax.text(5, .05, "η = taxa de aprendizado; cada árvore é ajustada ao gradiente da função de perda, com penalização Ω à complexidade.", ha="center", fontsize=7.8, style="italic")
fig.tight_layout(); fig.savefig("figuras/fig2_boosting.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------- Figura 3: divisão temporal ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
segs = [(0.3, 5.4, G, "Treinamento (60%)"), (5.7, 1.8, Y, "Validação (20%)"), (7.5, 2.0, O, "Teste (20%)")]
for x, w, c, t in segs:
    ax.add_patch(FancyBboxPatch((x, 2.3), w - .08, .7, boxstyle="round,pad=0,rounding_size=0.05", fc=c, ec="none", alpha=.9))
    ax.text(x + (w - .08) / 2, 2.65, t, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
arrow(ax, (.3, 2.1), (9.5, 2.1)); ax.text(9.55, 2.1, "tempo", va="center", fontsize=9)
ax.text(2.9, 1.65, "ajusta os\nparâmetros do modelo", ha="center", fontsize=8.5)
ax.text(6.55, 1.65, "escolhe os\nhiperparâmetros", ha="center", fontsize=8.5)
ax.text(8.5, 1.65, "avaliação final,\nfeita uma única vez", ha="center", fontsize=8.5)
ax.text(.3, 3.35, "2010", fontsize=8.5); ax.text(9.2, 3.35, "2025", fontsize=8.5)
# reajuste
ax.add_patch(FancyBboxPatch((.3, .35), 7.12, .6, boxstyle="round,pad=0,rounding_size=0.05", fc="none", ec=K, lw=1, ls="--"))
ax.text(3.9, .65, "Modelo final reajustado em treinamento + validação (hiperparâmetros fixos)", ha="center", va="center", fontsize=8.5)
arrow(ax, (7.42, .65), (8.4, 1.3))
fig.tight_layout(); fig.savefig("figuras/fig3_divisao_temporal.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("ok")
