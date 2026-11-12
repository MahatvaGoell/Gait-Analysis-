"""Write paper/main.tex (IEEE conference format, four authors in one row) from paper/content.py."""
import re
from pathlib import Path

import content as c

HERE = Path(__file__).resolve().parent


def tex(text):
    text = text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
    for a, b in (("\u2192", "$\\to$"), ("\u00b1", "$\\pm$"), ("\u03b3", "$\\gamma$"), ("\u2013", "--"), ("\u2014", "---"),
                 ("\u201c", "``"), ("\u201d", "''"), ("\u2019", "'")):
        text = text.replace(a, b)
    text = text.replace("\u221a*d*", "$\\sqrt{d}$")
    text = text.replace("{{KERNEL}}", "$K(\\mathbf{x},\\mathbf{x}')=\\exp(-\\gamma\\lVert\\mathbf{x}-\\mathbf{x}'\\rVert^2)$")
    text = re.sub(r"\*v\*_\{(\d+)\}", r"$v_{\1}$", text)
    text = re.sub(r"\*r\*", r"$r$", text)
    text = re.sub(r"\*C\* = 10", r"$C=10$", text)
    text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)
    text = re.sub(r"(\d) (cm|ms|Hz|s)\b", r"\1\\,\2", text)
    text = re.sub(r"\b(Fig\.|Table) ", lambda m: m.group(1) + "~", text)
    return text


LABELS = {"Fig. 1": "fig:flow", "Fig. 2": "fig:setup", "Fig. 3": "fig:seg", "Fig. 4": "fig:confusion",
          "Table I": "tab:t1", "Table II": "tab:t2", "Table III": "tab:t3"}

FLOW = r"""\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, minimum height=1.25cm, text width=2.35cm, fill=blue!6},
  lab/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, minimum height=0.85cm, text width=2.35cm, fill=orange!10},
  arr/.style={-{Stealth[length=2mm]}, thick}, node distance=0.42cm]
\node[box] (a) {\textbf{Data acquisition}\\8-channel thigh FMG band, 100\,Hz};
\node[box, right=of a] (b) {\textbf{Pre-processing}\\quiet-standing baseline removed};
\node[box, right=of b] (c) {\textbf{Segmentation}\\gait cycles, resampled to 100 points};
\node[box, right=of c] (d) {\textbf{Feature extraction}\\6 statistics + 10-point waveform per channel (128)};
\node[box, right=of d] (e) {\textbf{Classification}\\LDA, SVM, RF, HGB};
\node[box, right=of e] (f) {\textbf{Evaluation}\\LOSO and within-participant; accuracy, BA, F1, kappa};
\node[lab, below=0.45cm of a] (g) {Force-sensitive insoles (CoP)};
\node[lab, below=0.45cm of c] (h) {Labels: GI, short, long, S$\to$L, L$\to$S, GT};
\foreach \x/\y in {a/b,b/c,c/d,d/e,e/f} \draw[arr] (\x) -- (\y);
\draw[arr, dashed] (g) -- (h);
\draw[arr, dashed] (h) -- (c);
\draw[arr, dashed] (h.east) -| node[pos=0.25, below, font=\scriptsize]{training labels only} (e.south);
\end{tikzpicture}
\caption{%s}
\label{fig:flow}
\end{figure*}
"""
SETUP = r"""\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/participant_setup.png}\\[3pt]
\includegraphics[height=2.45cm]{figures/fsr_placement.png}\hfill
\raisebox{0.75cm}{\includegraphics[width=0.6\columnwidth]{figures/fmg_band.png}}
\caption{%s}
\label{fig:setup}
\end{figure}
"""
SIMPLE = r"""\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/%s}
\caption{%s}
\label{%s}
\end{figure}
"""


def cell(text, bold=False):
    t = tex(text)
    return "\\textbf{" + t + "}" if bold else t


T1 = (r"""\begin{table}[!t]
\caption{%s}
\label{tab:t1}
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcc}
\toprule
""" % tex(c.TABLE_TITLES["t1"]))
for r, row in enumerate(c.T1):
    if len(row) == 1:
        T1 += "\\multicolumn{4}{l}{" + tex(row[0][0]) + "} \\\\\n"
        continue
    T1 += " & ".join(cell(x, (r, k) in c.T1_BOLD) for k, x in enumerate(row)) + " \\\\\n"
    if r == 0:
        T1 += "\\midrule\n"
    if r == 8:
        T1 += "\\midrule\n"
T1 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

T2 = (r"""\begin{table}[!t]
\caption{%s}
\label{tab:t2}
\centering
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Unseen participants} & \multicolumn{3}{c}{Known participants} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
""" % tex(c.TABLE_TITLES["t2"]))
for row in c.T2[1:-1]:
    T2 += " & ".join(tex(x) for x in row) + " \\\\\n"
    if row[0] == "Class":
        T2 += "\\midrule\n"
T2 += "\\midrule\nAccuracy / kappa & \\multicolumn{3}{c}{62.5 / 0.49} & \\multicolumn{3}{c}{90.9 / 0.87} \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"

T3 = (r"""\begin{table}[!t]
\caption{%s}
\label{tab:t3}
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
""" % tex(c.TABLE_TITLES["t3"]))
for r, row in enumerate(c.T3):
    T3 += " & ".join(cell(x, (r, k) in c.T3_BOLD) for k, x in enumerate(row)) + " \\\\\n"
    if r == 0:
        T3 += "\\midrule\n"
T3 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

FIGS = {"flow": FLOW % tex(c.CAPTIONS["flow"]), "setup": SETUP % tex(c.CAPTIONS["setup"]),
        "seg": SIMPLE % ("fig1_segmentation.pdf", tex(c.CAPTIONS["seg"]), "fig:seg"),
        "confusion": SIMPLE % ("fig_confusion.pdf", tex(c.CAPTIONS["confusion"]), "fig:confusion")}
TABLES = {"t1": T1, "t2": T2, "t3": T3}

out = [r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{newtxtext,newtxmath}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}
\usepackage[hidelinks]{hyperref}
\raggedbottom

\begin{document}

\title{%s}

\author{
""" % c.TITLE]
blocks = []
for name, lines in c.AUTHORS:
    blocks.append("\\IEEEauthorblockN{%s}\n\\IEEEauthorblockA{\\small %s}" % (name, "\\\\\n".join(lines)))
out.append("\n\\and\n".join(blocks) + "\n}\n\n\\maketitle\n\\thispagestyle{plain}\n\\pagestyle{plain}\n\n")
out.append("\\begin{abstract}\n" + tex(c.ABSTRACT) + "\n\\end{abstract}\n\n")
out.append("\\begin{IEEEkeywords}\n" + tex(c.KEYWORDS) + "\n\\end{IEEEkeywords}\n\n")
for kind, value in c.BLOCKS:
    if kind == "h1":
        out.append("\\section{%s}\n" % value)
    elif kind == "h2":
        out.append("\\subsection{%s}\n" % value)
    elif kind == "p":
        out.append(tex(value) + "\n\n")
    elif kind == "bullet":
        out.append("\\begin{itemize}\n\\item " + tex(value) + "\n\\end{itemize}\n")
    elif kind == "fig":
        out.append(FIGS[value] + "\n")
    elif kind == "table":
        out.append(TABLES[value] + "\n")
src = "".join(out).replace("\\end{itemize}\n\\begin{itemize}\n", "")
out_refs = "\\begin{thebibliography}{00}\n" + "".join("\\bibitem{r%d} %s\n" % (i + 1, tex(r)) for i, r in enumerate(c.REFERENCES)) + "\\end{thebibliography}\n\n\\end{document}\n"
(HERE / "main.tex").write_text(src + out_refs, encoding="utf-8")
print("main.tex written")
