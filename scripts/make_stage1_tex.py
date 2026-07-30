#!/usr/bin/env python
"""Emit the Stage 1 LaTeX report and compile it."""
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path("/sessions/festive-awesome-hypatia/mnt/outputs")
SRC = ROOT / "rot/out/figures"
BUILD = ROOT / "work/tex"
BUILD.mkdir(parents=True, exist_ok=True)
(BUILD / "fig").mkdir(exist_ok=True)
for f in SRC.glob("*.pdf"):
    shutil.copy(f, BUILD / "fig" / f.name)

J = json.load(open(ROOT / "work/stage1.json"))
CFG = ["rot_mid_identity", "rot_flat_identity", "rot_mid", "rot_flat"]
ARCH = ["ATTN-M", "ATTN-S"]


def tt(s):
    return "\\texttt{" + str(s).replace("_", r"\_") + "}"


L = []
A = L.append

A(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\normalfont\Large\bfseries\color{blue!30!black}}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries\color{blue!35!black}}{\thesubsection}{1em}{}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\newcommand{\vs}{\sigma^2}

\title{\bfseries In-context unlearning:\\ membership AUC under prompt-time corruption\\
\large Stage 1 --- the AUC vs.\ $\mathrm{Var}(\epsilon)$ sweep}
\author{}
\date{30 July 2026}

\begin{document}
\maketitle
\vspace{-2.2em}

\begin{abstract}
\noindent
We measure whether corrupting a forget example \emph{in the prompt} removes the
membership evidence a shadow-model attacker can extract. Three corruption
families from the brief are swept across four decades of noise variance on four
configurations, with 512 shadow models per hypothesis and nine independent
seed combinations per point. Three results. First, the measured AUC agrees with
an independently derived closed form to $4\times10^{-4}$, which validates the
pipeline end to end. Second, the movement toward chance is almost entirely
\emph{masking} --- added variance --- rather than removal, at every corruption
strength below $\vs\approx1$. Third, and most consequential, reaching
chance costs a factor of ${\sim}2400$ in preservation error: within the swept
range there is no operating point at which the attacker is defeated and the
model is preserved. The arms also separate cleanly by which moment of the score
distribution they move: the zero-mean input-noise arm can only approach chance,
whereas the deterministic label flip carries a mean shift with no variance and
overshoots \emph{past} it. Finally, a negative control: equalising the
group covariance spectra does \emph{not} return the baseline AUC to chance,
so the signal is not an artefact of one group being intrinsically harder.
\end{abstract}
""")

# ---------------------------------------------------------------- 1 setup
A(r"""
\section{Setup}

A linear-attention model reads a query $x_q$, a retain set $\mathcal{D}_r$, and
a corrupted forget set $\widetilde{\mathcal{D}}_f$ in context. Targets are
linear in a task vector drawn afresh per sequence,
$y = \omega^{\!\top} x$, so the model must infer $\omega$ from the context
rather than memorise it. Because the model is linear in the context, the whole
ensemble reduces to one read-out matrix $M$ per shadow, and a shadow ensemble of
512 trains in seconds.

\paragraph{Corruption families.} The forget tokens are edited at prompt time,
with no weight update:
\begin{align*}
\text{C1} &: (x_f,\,y_f) \to (x_f,\ y_f+\epsilon), &&\epsilon\sim\mathcal{N}(0,\vs)\\
\text{C2} &: (x_f,\,y_f) \to (x_f+\epsilon,\ y_f), &&\epsilon\sim\mathcal{N}(0,\vs I_D)\\
\text{C3} &: (x_f,\,y_f) \to (x_f+\epsilon_1,\ y_f+\epsilon_2), &&\epsilon_1\perp\epsilon_2
\end{align*}
plus a deterministic label flip $y_f\to(1-2t)y_f$ and an input-whitening arm
that maps the forget covariance toward the retain covariance.

\paragraph{The attacker.} Membership inference over the \emph{model} axis at a
fixed probe point: $H_1$ is the fully trained model after the in-context edit,
$H_0$ is an oracle retrained without the forget group. The statistic is the
sign-aligned residual $\mathrm{sign}(y)\,(\hat y - y)$, ranked across the shadow
axis by Mann--Whitney and averaged over 64 probe points. Success is
$\mathrm{AUC}\to 0.5$.

\paragraph{Two conventions that matter.} Both hypotheses are scored on the
\emph{same} corrupted prompt with the same noise draw. Scoring the oracle on a
clean prompt instead lets a distinguisher win by detecting that an edit
occurred at all, and that confound grows with the very axis being plotted.
Separately, each point is repeated with a single noise draw broadcast across all
shadows; comparing against the per-shadow draw isolates how much of any AUC
movement is variance inflation rather than information removal.
""")

# ---------------------------------------------------------------- 2 configs
A(r"""
\section{Configurations}

All four configurations give the three groups the \emph{same} covariance
spectrum, so no group is intrinsically harder than another and the usual
difficulty confound is unavailable. They differ in how anisotropic that shared
spectrum is (participation ratio $\mathrm{PR}$, where $\mathrm{PR}=1$ is a
needle and $\mathrm{PR}=D=4$ a sphere) and in whether each group is rotated into
its own eigenbasis.
""")
rows = []
for c in CFG:
    m = J["meta"][c]
    rows.append(f"{tt(c)} & {m['PR'][0]:.2f} & {m['basis']} & {m['D']} & "
                f"{m['N']} & {m['S']} & {m['combos']} & {m['rows']} \\\\")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{lccccccc}
\toprule
configuration & PR (all groups) & basis & $D$ & $N$ & shadows & seed combos & rows \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{The four configurations. The two \emph{identity} rows are nulls: with
one spectrum and one basis the three groups are literally the same
distribution, so the full and oracle models train on indistinguishable data and
the AUC must be chance. They are a correctness test on the pipeline, not an
experiment.}
\end{table}
""")

# ---------------------------------------------------------------- 3 nulls
b = J["baseline"]
A(r"""
\section{Results}
\subsection{The nulls pass}
""")
rows = []
for c in CFG:
    r = b[c]
    tag = r"\textbf{" if not c.endswith("identity") else ""
    end = "}" if tag else ""
    rows.append(f"{tt(c)} & {tag}{r['ATTN-M']['res']:.4f}{end} & "
                f"{r['ATTN-M']['lo']:.4f}--{r['ATTN-M']['hi']:.4f} & "
                f"{tag}{r['ATTN-S']['res']:.4f}{end} & "
                f"{r['ATTN-S']['lo']:.4f}--{r['ATTN-S']['hi']:.4f} \\\\")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{ATTN-M} & \multicolumn{2}{c}{ATTN-S}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
configuration & AUC & across-seed range & AUC & across-seed range \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Membership AUC at zero corruption. The two null configurations sit at
chance, confirming that the two hypotheses are not coupled anywhere in the
training or evaluation code.}
\end{table}

\subsection{The baseline signal survives equalised spectra}

The two rotated configurations are the result, and they are a negative one. All
three groups share a spectrum, so every group is equally hard; yet the AUC sits
0.07--0.08 from chance. The across-seed range is $0.0009$ wide for
""" + tt("rot_mid") + r""" ATTN-M --- three independent training runs agreeing to
the fourth decimal. This is systematic, not sampling noise.

This was predicted to go the other way. The expectation recorded before the run
was that equalising the spectra would largely cancel the read-out mismatch
between the two hypotheses and drive AUC toward chance. It does not, and the
prediction is recorded as failed. A simulation containing \emph{no} memorisation
--- both models replaced by the idealised preconditioner
$\Lambda_{\text{train}}^{-1}$ for the groups each trained on --- predicts
$0.313$ for """ + tt("rot_mid") + r""" against $0.417$ measured: the right sign
and rough magnitude, over-predicting the confound. Decomposing the residual
into read-out mismatch versus genuine memorisation is the main open question
and requires the read-out covectors from the trained ensembles.
""")

# ---------------------------------------------------------------- 4 sweeps
A(r"""
\subsection{The sweeps}

\begin{figure}[h]\centering
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C1_rot_mid.pdf}
\caption{C1, label noise}\end{subfigure}\hfill
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C2_rot_mid.pdf}
\caption{C2, input noise}\end{subfigure}\\[4pt]
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C3_rot_mid.pdf}
\caption{C3, both}\end{subfigure}\hfill
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_flip_rot_mid.pdf}
\caption{deterministic label flip}\end{subfigure}
\caption{Membership AUC against corruption strength on """ + tt("rot_mid") + r""".
Bands are the min--max range across nine (training seed, probe seed)
combinations, not a within-run bootstrap. The thin dark line under C1 and C2 is
the closed-form prediction. The three Gaussian arms rise toward chance and stop;
the deterministic flip (d) rises \emph{through} chance and keeps going. Section
3.6 explains why.}
\end{figure}

\begin{figure}[h]\centering
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C1_rot_flat.pdf}
\caption{C1 on """ + tt("rot_flat") + r"""}\end{subfigure}\hfill
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C2_rot_flat.pdf}
\caption{C2 on """ + tt("rot_flat") + r"""}\end{subfigure}\\[4pt]
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C1_rot_mid_identity.pdf}
\caption{C1 on the null}\end{subfigure}\hfill
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/auc_C2_rot_mid_identity.pdf}
\caption{C2 on the null}\end{subfigure}
\caption{The same sweeps on the flatter configuration and on a null. The null
panels are flat at chance across the entire grid, which is what a corruption
sweep should look like when there is nothing to remove.}
\end{figure}
""")

# ---------------------------------------------------------------- 5 theory
A(r"""
\clearpage
\subsection{The closed form is exact}

Because the read-out is linear in the context, the effect of Gaussian context
noise is available analytically. Writing $c = M^{\!\top}t_q = [c_x; c_y]$ and
$u=\sum_i t_i y_i$ with $t_i=[x_i;y_i]$, the last coordinate of $u$ is
$\sum_i y_i^2$ --- the label enters \emph{quadratically}. Hence
\[
\text{C1}:\quad \mathbb{E}[\Delta\hat y] = \frac{c_y\, n_f\, \vs}{N+1},
\qquad
\text{C2}:\quad \mathbb{E}[\Delta\hat y] = 0 .
\]
C2 is exactly zero-mean; C1 is not, because the $\epsilon^2$ term survives
expectation in the label--label block. Feeding these moments through a Gaussian
approximation of the score distribution gives a predicted AUC with no free
parameters.
""")
rows = []
for c in ["rot_mid", "rot_flat"]:
    for k, v in J["theory"][c].items():
        a, arm = k.split("|")
        rows.append(f"{tt(c)} & {a} & {arm} & {v['mean']:.5f} & {v['max']:.5f} \\\\")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{lllcc}
\toprule
configuration & architecture & arm & mean $|$error$|$ & max $|$error$|$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Absolute difference between the predicted and measured AUC over the
whole grid. Agreement at $4\times10^{-4}$ means the simulation and the algebra
are independently confirming each other. A systematic gap would have localised a
bug in one of them.}
\end{table}

""")

# ------------------------------------------------------- 3.6 channel separation
rowsx = []
for c in ["rot_mid", "rot_flat"]:
    for a in ARCH:
        for arm in ["C1", "C2", "C3", "flip", "whiten"]:
            s = J["sweep"][c][a].get(arm)
            if not s:
                continue
            mn, mx = min(s["auc"]), max(s["auc"])
            cross = mx > 0.5005
            where = ""
            if cross:
                where = f"{min(x for x, v in zip(s['x'], s['auc']) if v > 0.5):g}"
            rowsx.append(f"{tt(c)} & {a} & {arm} & {mn:.4f} & {mx:.4f} & "
                         + (r"\textbf{yes}" if cross else "no") + f" & {where} \\\\")
A(r"""
\subsection{The two channels, separated}

The arms differ in \emph{which} moment of the score distribution they move, and
the sweeps separate the two cleanly.
""")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{lllcccc}
\toprule
configuration & arch & arm & min AUC & max AUC & crosses $0.5$? & at \\
\midrule
""" + "\n".join(rowsx) + r"""
\bottomrule
\end{tabular}
\caption{Range of the AUC curve for every arm. Only the deterministic flip
crosses chance, and it does so in all four cases.}
\end{table}

The pattern follows directly from the algebra. Write the predicted AUC as
$\Phi\!\big((\mu_1-\mu_0)/\sqrt{v_1+v_0}\big)$ and note that corruption can move
the numerator, the denominator, or both:

\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
\item \textbf{C2} is \emph{exactly} zero-mean. It can only inflate the
denominator, so it drives AUC toward $0.5$ and mathematically cannot cross.
\item \textbf{flip} is deterministic given the probe: it carries a mean shift
and \emph{no} variance at all. The numerator moves while the denominator stands
still, so it must eventually cross --- and it does, at $t\approx0.9$--$1.0$,
reaching $0.533$.
\item \textbf{C1} carries both. Its mean shift grows like $\vs$ while its
spread grows like $\sqrt{\vs}$, so it was predicted to cross as well. It does
not: the variance wins over the whole swept range, and the maximum reached is
$0.4974$. The mean term is real and is included in the closed form --- which
still matches to $4\times10^{-4}$ --- so the algebra is not at fault; the
coefficient $c_y$ in the trained models is simply too small.
\end{itemize}

Crossing chance is not success. An AUC of $0.533$ is a \emph{worse} outcome than
$0.49$: the attacker is again distinguishing the two hypotheses, merely with the
ranking inverted. The flip arm therefore overshoots the target rather than
reaching it, which is exactly the failure mode a na\"ive $\min\mathrm{AUC}$
objective would drive toward.

\paragraph{The whitening arm barely moves.} Mapping the forget inputs toward the
retain covariance shifts AUC only from $0.4273$ to $0.4394$ at full strength on
""" + tt("rot_mid") + r""" --- about $15\%$ of the distance to chance. Since all
three groups share a spectrum in these configurations, there is little for a
covariance-matching edit to change, so this is expected here and should be
re-checked on the base configuration where the spectra genuinely differ.
""")

# ---------------------------------------------------------------- 6 masking
A(r"""
\subsection{The movement toward chance is masking, not removal}

Per-shadow corruption noise inflates within-ensemble spread, which lowers AUC
mechanically without removing anything. Repeating each point with one noise draw
shared across all shadows holds that channel fixed. The difference is the
portion of the movement attributable to masking.
""")
rows = []
for c in ["rot_mid", "rot_flat"]:
    for k in ["ATTN-M|C1", "ATTN-M|C2", "ATTN-M|C3"]:
        m = J["masking"][c][k]
        vals = " & ".join(f"{m[x]:.0f}\\%" if m.get(x) is not None else "---"
                          for x in ["0.01", "0.1", "1.0", "10.0", "100.0"])
        rows.append(f"{tt(c)} & {k.split('|')[1]} & {vals} \\\\")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{llccccc}
\toprule
configuration & arm & $\vs{=}0.01$ & $0.1$ & $1$ & $10$ & $100$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Percentage of the movement toward chance attributable to variance
inflation rather than information removal (ATTN-M). Below $\vs\approx1$ both
Gaussian arms are essentially pure masking. Genuine removal only takes over at
large corruption, and sooner for the label arm than the input arm.}
\end{table}

This has a direct consequence for how any result here should be stated. At small
corruption strengths the attacker is defeated by added spread, not because the
forget information has gone --- an attacker who could average away the spread
would recover the signal.
""")

# ---------------------------------------------------------------- 7 tradeoff
A(r"""
\subsection{Removal costs preservation by a factor of ${\sim}2400$}

Reaching chance is not by itself a result: one can always reach chance by
destroying the context. The question is whether an operating point exists where
the attacker is defeated \emph{and} the model is preserved. We measure
preservation as $\varepsilon = \mathrm{KL}(p_{\text{oracle}}\,\|\,p_{\text{unlearned}})$
on the forget-population residual law, and compare the corruption strength that
minimises $\varepsilon$ against the strength that brings AUC closest to chance.
""")
rows = []
for c in ["rot_mid", "rot_flat"]:
    for k in ["ATTN-M|C1", "ATTN-M|C2", "ATTN-M|C3",
              "ATTN-S|C1", "ATTN-S|C2", "ATTN-S|C3"]:
        t = J["tradeoff"][c][k]
        a, arm = k.split("|")
        rows.append(f"{tt(c)} & {a} & {arm} & {t['sig_eps']:g} & {t['eps_min']:.5f} "
                    f"& {t['sig_auc']:g} & {t['auc_best']:.4f} & {t['eps_at_auc']:.3f} "
                    f"& $\\mathbf{{{t['ratio']:.0f}\\times}}$ \\\\")
A(r"""
\begin{table}[h]\centering\small
\begin{tabular}{lllcccccc}
\toprule
& & & \multicolumn{2}{c}{best preservation} & \multicolumn{3}{c}{closest to chance} & \\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}
config & arch & arm & $\vs$ & $\varepsilon_{\min}$ & $\vs$ & AUC & $\varepsilon$ & cost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{The removal/preservation tradeoff. The final column is
$\varepsilon$ at the chance-level operating point divided by
$\varepsilon_{\min}$. A ratio near 1 would mean a clean operating point exists.}
\end{table}

\begin{figure}[h]\centering
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/tradeoff_C1_rot_mid.pdf}
\caption{C1}\end{subfigure}\hfill
\begin{subfigure}{0.49\textwidth}\includegraphics[width=\linewidth]{fig/tradeoff_C2_rot_mid.pdf}
\caption{C2}\end{subfigure}
\caption{AUC (solid, left axis) against $\varepsilon$ (dashed, right axis, log
scale) on """ + tt("rot_mid") + r""". The two vertical markers are the
$\varepsilon$ minimum and the AUC-closest-to-chance point. They are three
decades apart.}
\end{figure}

\textbf{This is the main result of Stage 1.} The ratio is not close to one; it is
of order $2400$. Preservation error is minimised at or near zero corruption,
where AUC sits at $0.427$, and AUC only approaches chance at the top of the
swept range, by which point $\varepsilon$ has grown from $5\times10^{-4}$ to
above $1.2$. Within four decades of corruption strength there is no setting at
which the attacker is defeated and the model still behaves like the retrain
oracle. Combined with the masking result above, the honest reading is that these
corruption families trade removal against preservation rather than achieving
both.
""")

# ---------------------------------------------------------------- 8 caveats
A(r"""
\clearpage
\section{Corrections and caveats}

\paragraph{The masking sign convention inverts here.} The diagnostic is
documented as ``positive means part of the AUC \emph{drop} is masking'', which
assumes AUC starts above chance and falls. Every configuration measured starts
\emph{below} chance and rises, so every raw masking value is negative and the
documented reading inverts. The percentages in Table~3 correct for this; the
underlying numbers were never wrong, only the label.

\paragraph{AUC below $0.5$ is not a defect.} The statistic is a fixed-direction
Mann--Whitney rank on a chosen observable, not a likelihood-ratio test, so it is
free to land either side of chance. What matters is the distance from $0.5$.
A consequence worth stating: a one-sided statistic reaching $0.5$ does
\emph{not} establish indistinguishability, because a two-sided attacker could
still detect a variance difference. Whether it does here is measurable from the
cached ensembles and has not yet been checked.

\paragraph{The null test was initially wrong.} Its first version required every
per-row confidence interval at zero corruption to cover $0.5$. That fails on
correct data for three compounding reasons: at zero strength all corruption
modes are the identity edit, so one measurement was counted six times;
requiring 108 simultaneous 95\% intervals all to cover has probability
$0.95^{108}=0.4\%$ under a perfect null; and the nine seed combinations reuse
three trained ensembles and so are not independent. The replacement
deduplicates, clusters on the training seed, and adds an absolute bound because
a three-seed $t$-test has almost no power.

\paragraph{What has not been run.} These results are from four control
configurations. The base configuration --- three groups with \emph{differing}
spectra, which is the setting the brief describes most directly --- has not been
re-run since the pipeline fixes, and its numbers are not included here. A
stochastic Bernoulli label-flip arm has been implemented and validated against
Monte Carlo but not yet swept. Neither is expensive: both are minutes of GPU
time against cached or quickly retrained ensembles.
""")

A(r"\end{document}")

tex = BUILD / "stage1_report.tex"
tex.write_text("\n".join(L))
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                       cwd=BUILD, capture_output=True, text=True)
pdf = BUILD / "stage1_report.pdf"
if pdf.exists():
    shutil.copy(pdf, ROOT / "stage1_report.pdf")
    print("OK ->", ROOT / "stage1_report.pdf", pdf.stat().st_size, "bytes")
else:
    print(r.stdout[-3000:])
