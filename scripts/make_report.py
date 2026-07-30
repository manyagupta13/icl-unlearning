#!/usr/bin/env python
"""Build the two-stage report PDF."""
import json
import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = pathlib.Path("/sessions/festive-awesome-hypatia/mnt/outputs")
FIG = ROOT / "rot/out/figures"
NUM = json.load(open(ROOT / "work/numbers.json"))
OUT = ROOT / "icl_unlearning_report.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=16,
                    spaceAfter=7, textColor=colors.HexColor("#12305e"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=11,
                    spaceAfter=4, textColor=colors.HexColor("#28466f"))
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9.4, leading=13.4,
                      alignment=TA_JUSTIFY, spaceAfter=6)
MONO = ParagraphStyle("MONO", parent=ss["BodyText"], fontName="Courier",
                      fontSize=8.2, leading=11, leftIndent=12, spaceAfter=7,
                      textColor=colors.HexColor("#1a1a1a"))
CAP = ParagraphStyle("CAP", parent=ss["BodyText"], fontSize=8.2, leading=10.5,
                     textColor=colors.HexColor("#555555"), spaceAfter=9)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontSize=19, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5,
                     textColor=colors.HexColor("#555555"), spaceAfter=2)

S = []
P = lambda t, s=BODY: S.append(Paragraph(t, s))
GAP = lambda h=5: S.append(Spacer(1, h))


def table(data, widths, hi_rows=(), align_right_from=1):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3),
          ("FONT", (0, 1), (-1, -1), "Helvetica", 8.3),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12305e")),
          ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#12305e")),
          ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dddddd")),
          ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in hi_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef3fa")))
        st.append(("FONT", (0, r), (-1, r), "Helvetica-Bold", 8.3))
    t.setStyle(TableStyle(st))
    return t


# ============================================================== title
P("In-context unlearning: membership AUC under corruption", TITLE)
P("Stage 1 (AUC vs Var(eps) sweep) and Stage 2 (learned corruption policy)", SUB)
P("Working report &mdash; 30 July 2026", SUB)
GAP(10)

P("<b>Summary.</b> Stage 1 is complete and validated: the three corruption "
  "families from the brief are implemented, swept, and the measured AUC agrees "
  "with an independently derived closed form to 4x10<super>-4</super>. Four "
  "control configurations were run; two are exact nulls and both pass. The "
  "headline finding is negative and reproducible: the membership signal at zero "
  "corruption does <i>not</i> vanish when the group spectra are equalised, so it "
  "is not an artefact of one group being intrinsically harder. Stage 2 is "
  "implemented and demonstrated on a synthetic instance. Its main result is that "
  "the score-function (REINFORCE) estimator specified in the brief is not needed "
  "for the parameterisation the brief itself writes down &mdash; the objective is "
  "available in closed form &mdash; and that the objective as written minimises "
  "the wrong quantity.")

# ============================================================== 1
P("1. The brief", H1)
P("The handwritten notes specify two stages.", BODY)
P("<b>Stage 1.</b> A model reads a query x<sub>q</sub>, a retain set D<sub>r</sub>, "
  "and a corrupted forget set in context. Targets are linear, "
  "y<sub>f</sub> = w<super>T</super>x<sub>f</sub> and "
  "y<sub>r</sub> = w<super>T</super>x<sub>r</sub>. The forget point is corrupted "
  "three ways &mdash; (x<sub>f</sub>, y<sub>f</sub>+eps), (x<sub>f</sub>+eps, "
  "y<sub>f</sub>), and both &mdash; and the deliverable is membership AUC plotted "
  "against Var(eps).")
P("<b>Stage 2 (&ldquo;Generalize&rdquo;).</b> Replace the hand-designed corruption "
  "with a learned one, p<sub>theta</sub>(x&#732;<sub>f</sub>, y&#732;<sub>f</sub> | "
  "x<sub>f</sub>, y<sub>f</sub>), factorised as a Bernoulli(theta) label flip with "
  "the features passed through unchanged. Optimise "
  "min<sub>theta</sub> AUC(theta) + lambda||theta||. Because AUC is not "
  "differentiable through the sampling, the notes derive a score-function "
  "gradient, grad E[AUC] = E[grad log Q<sub>theta</sub> . AUC].")

# ============================================================== 2
P("2. Stage 1: implementation", H1)
P("The three corruption arms map one-to-one onto the codebase as C1 (label "
  "noise), C2 (input noise) and C3 (both), swept over 21 log-spaced points across "
  "four decades. Each sweep point is evaluated on frozen shadow ensembles &mdash; "
  "512 shadow models per hypothesis, trained once and reused, since a prompt-time "
  "edit never touches training. Three training seeds are crossed with three probe "
  "seeds, giving nine independent estimates per point.", BODY)
P("Beyond the brief, four things were added because they change what the numbers "
  "mean:", BODY)
P("<b>Matched-context comparison.</b> Both hypotheses are scored on the <i>same</i> "
  "corrupted prompt. Scoring the retrain oracle on a clean prompt lets a "
  "distinguisher win by detecting that an edit happened at all, and that confound "
  "grows with the very axis being plotted.", BODY)
P("<b>A masking control.</b> Per-shadow corruption noise inflates within-ensemble "
  "spread, which lowers AUC mechanically without removing anything. Repeating each "
  "point with one noise draw broadcast across shadows separates that from genuine "
  "removal.", BODY)
P("<b>A closed-form overlay.</b> The model is linear in the context, so the effect "
  "of Gaussian context noise on the read-out is available analytically. Drawing it "
  "under the measurements turns the figure into a standing correctness check.", BODY)
P("<b>Null controls.</b> Two configurations give all three groups the same "
  "covariance spectrum <i>and</i> the same basis, making them one distribution. "
  "The full and oracle models then train on indistinguishable data, so AUC must be "
  "chance. If it is not, the two hypotheses are coupled somewhere in the code and "
  "nothing else is interpretable.", BODY)

# ============================================================== 3
P("3. Stage 1: results", H1)
P("3.1 Baseline membership signal", H2)
P("AUC at zero corruption, on the sign-aligned residual. Values are means over "
  "three training seeds; the range is the spread of the three per-seed means.", BODY)
rows = [["configuration", "groups", "ATTN-M", "range", "ATTN-S", "range"]]
lab = {"rot_mid_identity": ("rot_mid_identity", "identical"),
       "rot_mid": ("rot_mid", "rotated, PR 2.65"),
       "rot_flat_identity": ("rot_flat_identity", "identical"),
       "rot_flat": ("rot_flat", "rotated, PR 3.55")}
byc = {}
for c, a, m, lo, hi in NUM["baseline"]:
    byc.setdefault(c, {})[a] = (m, lo, hi)
for c in ["rot_mid_identity", "rot_flat_identity", "rot_mid", "rot_flat"]:
    m = byc[c]["ATTN-M"]; s = byc[c]["ATTN-S"]
    rows.append([lab[c][0], lab[c][1], f"{m[0]:.4f}", f"{m[1]:.4f}-{m[2]:.4f}",
                 f"{s[0]:.4f}", f"{s[1]:.4f}-{s[2]:.4f}"])
S.append(table(rows, [3.3*cm, 2.9*cm, 1.9*cm, 2.6*cm, 1.9*cm, 2.6*cm],
               hi_rows=(3, 4)))
GAP(3)
P("The two <i>identity</i> rows are the nulls and both sit at chance, which "
  "validates the pipeline. The two rotated rows are the result. All three groups "
  "there share one spectrum, so every group is equally hard and no "
  "difficulty confound is available &mdash; yet AUC sits 0.07 to 0.08 away from "
  "chance, and reproduces across independent training seeds to within 0.001. This "
  "is a systematic effect, not sampling noise.", BODY)
P("This was predicted to go the other way. The expectation recorded before the run "
  "was that equalising the spectra would largely cancel the read-out mismatch "
  "between the two hypotheses and push AUC toward 0.5. It does not. The prediction "
  "is recorded as failed.", BODY)

P("3.2 The closed form is exact", H2)
rows = [["configuration", "arch", "arm", "mean |error|", "max |error|"]]
for c, a, m, mean, mx in NUM["theory"]:
    rows.append([c, a, m, f"{mean:.5f}", f"{mx:.5f}"])
S.append(table(rows, [3.0*cm, 2.0*cm, 1.4*cm, 2.6*cm, 2.6*cm]))
GAP(3)
P("Absolute difference between the analytically predicted AUC and the measured "
  "AUC, across the whole corruption grid. Agreement at 4x10<super>-4</super> means "
  "the simulation pipeline and the algebra are independently confirming each "
  "other; a systematic gap would have indicated a bug in one of them.", CAP)

S.append(PageBreak())
P("3.3 The corruption sweeps", H2)
for f, cap in ((FIG / "auc_C1_rot_mid.png",
                "C1, additive label noise, on rot_mid. Both architectures rise "
                "monotonically from the baseline toward chance and stop there. "
                "The thin dark line is the closed-form prediction; the band is "
                "the spread across nine seed combinations."),
               (FIG / "auc_C2_rot_mid.png",
                "C2, additive input noise, same configuration. Approaches chance "
                "faster than C1 at matched noise variance.")):
    if f.exists():
        S.append(Image(str(f), width=11.2*cm, height=9.1*cm, kind="proportional"))
        P(cap, CAP)

P("3.4 Neither arm crosses chance", H2)
rows = [["configuration", "arch", "arm", "min AUC", "max AUC", "crosses 0.5?"]]
for c, a, m, mn, mx in NUM["nocross"]:
    rows.append([c, a, m, f"{mn:.4f}", f"{mx:.4f}", "no"])
S.append(table(rows, [2.7*cm, 1.9*cm, 1.3*cm, 2.1*cm, 2.1*cm, 2.3*cm]))
GAP(3)
P("A second pre-registered prediction, also failed. The label-noise arm carries a "
  "mean shift that grows linearly in the noise variance while the spread grows "
  "only as its square root, so it was predicted to overshoot chance and keep "
  "going. It does not: across four decades the maximum reached is 0.4989. The "
  "mean-shift term is real and is included in the closed form, which still matches "
  "to 4x10<super>-4</super>, so the algebra is not at fault &mdash; the "
  "coefficient is simply too small in the trained models to matter over this "
  "range.", BODY)

P("3.5 Masking versus removal", H2)
P("Percentage of the movement toward chance attributable to variance inflation "
  "rather than to information removal, on rot_mid, ATTN-M.", BODY)
rows = [["arm", "s2=0.01", "s2=0.1", "s2=1", "s2=10", "s2=100"]]
for r in NUM["masking"]:
    rows.append([r[0]] + [f"{v:.0f}%" for v in r[1:]])
S.append(table(rows, [1.6*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm]))
GAP(3)
P("Below roughly unit variance both arms are essentially pure masking: the "
  "attacker is defeated by added spread, not because the forget information has "
  "gone. Genuine removal only takes over at large corruption, and much sooner for "
  "the label arm than the input arm. This matters for any claim that a corruption "
  "&ldquo;worked&rdquo;: at small corruption strengths it has not removed "
  "anything.", BODY)

S.append(PageBreak())
# ============================================================== 4
P("4. Corrections made along the way", H1)
P("<b>The flip arm did not match the brief.</b> The existing label-flip "
  "implementation was deterministic, y<sub>f</sub> -&gt; (1-2t)y<sub>f</sub>, "
  "whereas the brief specifies a Bernoulli(theta) flip. These share a mean, "
  "E[(1-2B)y] = (1-2theta)y, but the Bernoulli version also carries variance "
  "4theta(1-theta)y<super>2</super>. Given that most of the measured effect at "
  "small corruption is masking, a corruption family with no variance channel "
  "cannot exhibit the dominant mechanism. A correct Bernoulli arm was added; "
  "section 5 derives it.", BODY)
P("<b>The masking diagnostic's sign convention is inverted here.</b> It is "
  "documented as &ldquo;positive means part of the AUC <i>drop</i> is masking&rdquo;, "
  "which assumes AUC starts above chance and falls. Every configuration measured "
  "starts <i>below</i> chance and rises, so every masking value is negative and "
  "the documented reading inverts. The numbers are correct; the label is "
  "misleading.", BODY)
P("<b>Figure filenames collided across configurations.</b> Two plotting scripts "
  "wrote names that omitted the configuration, so a multi-configuration batch "
  "silently overwrote its own output. Fixed and covered by a test.", BODY)
P("<b>The null-control test was statistically invalid.</b> Its first version "
  "required every per-row confidence interval at zero corruption to cover 0.5. "
  "That fails on correct data for three compounding reasons: at zero strength all "
  "six corruption modes are the identity edit, so one measurement was counted six "
  "times; requiring 108 simultaneous 95% intervals all to cover has probability "
  "0.95<super>108</super> = 0.4% under a perfect null; and the nine seed "
  "combinations reuse three trained ensembles, so they are not independent. The "
  "replacement deduplicates, clusters on the training seed, and adds an absolute "
  "bound because a three-seed t-test has almost no power.", BODY)

# ============================================================== 5
P("5. The Bernoulli arm", H1)
P("With y&#732;<sub>i</sub> = (1-2B<sub>i</sub>)y<sub>i</sub> and "
  "B<sub>i</sub> ~ Bernoulli(theta), write a<sub>i</sub> = y<sub>i</sub>(c<sub>x</sub> . "
  "x<sub>i</sub>) over the forget slice, where c<sub>x</sub> is the input block of "
  "the read-out covector. Then", BODY)
P("E[delta yhat]&nbsp;&nbsp;= -2 theta / (N+1) . sum a<sub>i</sub><br/>"
  "Var[delta yhat] = 4 theta (1-theta) / (N+1)<super>2</super> . "
  "sum a<sub>i</sub><super>2</super>", MONO)
P("The step that makes this cleaner than the additive label-noise arm is that "
  "(1-2B)<super>2</super> = 1 identically, so a sign flip leaves the label-label "
  "block of the context vector <i>exactly</i> unchanged. The quadratic drift that "
  "complicates C1 has no analogue here, and the label block of the read-out never "
  "enters. Verified at 400,000 trials per point:", BODY)
rows = [["theta", "E[d yhat] measured", "predicted", "Var measured", "predicted"]]
for th, em, tm, ev, tv in [("0.05", "-0.001209", "-0.001213", "3.994e-05", "4.002e-05"),
                           ("0.10", "-0.002444", "-0.002426", "7.620e-05", "7.582e-05"),
                           ("0.25", "-0.006076", "-0.006065", "1.583e-04", "1.580e-04"),
                           ("0.50", "-0.012126", "-0.012131", "2.108e-04", "2.106e-04"),
                           ("0.75", "-0.018210", "-0.018196", "1.579e-04", "1.580e-04"),
                           ("1.00", "-0.024262", "-0.024262", "0 (exact)", "0")]:
    rows.append([th, em, tm, ev, tv])
S.append(table(rows, [1.5*cm, 3.4*cm, 2.6*cm, 2.9*cm, 2.6*cm]))
GAP(3)
P("The variance is non-monotone in theta: zero at 0, maximal at 0.5, zero again at "
  "1 where the flip becomes deterministic. So the Bernoulli arm and the "
  "deterministic arm coincide at theta = 1 and differ most at theta = 0.5.", CAP)

S.append(PageBreak())
# ============================================================== 6
P("6. Stage 2: the learned corruption", H1)
P("6.1 The objective minimises the wrong quantity", H2)
P("The brief writes min<sub>theta</sub> AUC(theta) + lambda||theta||. Every "
  "configuration measured in Stage 1 has AUC <i>below</i> 0.5 and rising toward "
  "it. Minimising AUC therefore drives the corruption toward greater "
  "distinguishability: an AUC of 0.30 is a better attacker than 0.50, merely an "
  "inverted one. The success criterion is chance, so the objective must penalise "
  "distance from it:", BODY)
P("min<sub>theta</sub>  (AUC(theta) - 1/2)<super>2</super> + lambda ||theta||", MONO)
P("Both forms are implemented; the corrected one is the default. This is not a "
  "quibble about sign conventions &mdash; with the literal objective and a "
  "baseline of 0.417, gradient descent moves theta the wrong way.", BODY)

P("6.2 The score-function gradient is unnecessary here", H2)
P("The brief derives a REINFORCE estimator because AUC is not differentiable "
  "through the sampling of the corrupted labels. That reasoning is sound in "
  "general. But for the parameterisation the brief itself specifies &mdash; "
  "independent per-token Bernoulli flips with features passed through unchanged "
  "&mdash; the expectation can be taken analytically <i>before</i> any sampling. "
  "The two moments above are exact for a per-token theta<sub>i</sub> as well as a "
  "scalar theta, so this covers both the single-parameter form and the neural "
  "policy. Feeding them into the Gaussian AUC makes the objective an ordinary "
  "differentiable function.", BODY)
P("Measured on a synthetic instance matched to the shipped geometry (D=4, N=31, "
  "11 forget tokens, 64 probe points, 512 shadows), gradient of the corrected "
  "objective at theta = 0.15:", BODY)
rows = [["estimator", "gradient", "std. dev.", "cost per gradient", "noise / signal"],
        ["closed form", "-7.83e-03", "0 (exact)", "6 ms", "0"],
        ["REINFORCE, 8 samples", "-2.65e-02", "8.07e-02", "43 ms", "10.3"],
        ["REINFORCE, 32 samples", "+1.14e-04", "4.09e-02", "176 ms", "5.2"],
        ["REINFORCE, 128 samples", "-8.31e-03", "2.04e-02", "810 ms", "2.6"]]
S.append(table(rows, [4.0*cm, 2.5*cm, 2.3*cm, 3.0*cm, 2.6*cm], hi_rows=(1,)))
GAP(3)
P("The score-function estimator is unbiased and does converge on the closed-form "
  "value as the sample count grows, which confirms the two are the same objective. "
  "But at 128 samples it still costs 135 times more per gradient and its standard "
  "deviation is 2.6 times the quantity being estimated.", CAP)
P("REINFORCE remains necessary if the policy is extended to perturb the "
  "<i>features</i>, since it is precisely the delta(x<sub>f</sub>) factor that "
  "removes the cross terms. That path is kept in the implementation for that "
  "reason.", BODY)

P("6.3 The optimal policy, and a limit on the closed form", H2)
P("Optimising the corrected objective drives theta to 0.957, at which point the "
  "closed form reports AUC = 0.5000 exactly. The optimum is not a small "
  "perturbation: it flips almost every forget label. That is the mean-compensating "
  "solution &mdash; the flip's mean shift is used to <i>cancel</i> the "
  "pre-existing membership gap, not to avoid creating one. A corruption "
  "constrained to be mean-zero cannot reach chance at all.", BODY)
P("The closed form should not be trusted for the last step, however. Measured at "
  "the same theta, AUC is 0.514, not 0.500 &mdash; a residual gap of 0.014. The "
  "reason is that the corruption itself is not Gaussian: the shift is a sum of "
  "only eleven Bernoulli terms, and near the optimum that sum has skew +2.3 and "
  "excess kurtosis +4.9. This is a genuine difference from the additive-noise "
  "arms, where the noise is Gaussian by construction and the same closed form is "
  "accurate to 4x10<super>-4</super>.", BODY)
rows = [["theta", "AUC, closed form", "AUC, measured", "|measured - 0.5|"],
        ["0.000 (baseline)", "0.41421", "0.41414", "0.08586"],
        ["0.800", "0.49100", "0.50540", "0.00540"],
        ["0.900", "0.49418", "0.51328", "0.01328"],
        ["0.957 (closed-form optimum)", "0.50000", "0.51381", "0.01381"],
        ["1.000", "0.51139", "0.51157", "0.01157"]]
S.append(KeepTogether([
    table(rows, [5.6*cm, 3.2*cm, 3.0*cm, 3.0*cm], hi_rows=(2, 4)),
    Spacer(1, 3),
    Paragraph("The practical recipe this supports: optimise on the closed form, "
              "which is deterministic and roughly a hundred times cheaper per "
              "gradient and lands in the right neighbourhood in one shot, then "
              "polish against the measured AUC. Do not report the closed-form "
              "value as the result for this arm.", BODY)]))

# ============================================================== 7
P("7. Status", H1)
P("7.1 Complete", H2)
P("Stage 1 sweeps for all corruption arms, on four control configurations, with "
  "nine seed combinations each; the closed-form overlay and its validation; the "
  "Bernoulli arm and its Monte-Carlo validation; the Stage 2 policy, both "
  "gradient estimators, and the objective correction; and a synthetic-instance "
  "demonstration of the whole Stage 2 loop.", BODY)
P("7.2 Not yet run", H2)
P("The Stage 2 optimisation has not been run against the <i>trained</i> ensembles "
  "&mdash; only against a synthetic instance built to match their geometry. The "
  "script for it is written and reuses the cached ensembles, so it costs seconds "
  "of GPU time, not hours. The Bernoulli sweep has likewise not been run on the "
  "trained ensembles.", BODY)
P("Two questions remain open and both are cheap to settle from the cached "
  "ensembles. First, the decomposition of the residual 0.07 baseline signal into "
  "read-out mismatch versus genuine memorisation. Second, whether a two-sided "
  "attacker &mdash; one that detects a <i>variance</i> difference rather than a "
  "mean difference &mdash; still succeeds where the one-sided statistic reports "
  "chance. If it does, then &ldquo;AUC reaches 0.5&rdquo; overstates what has been "
  "achieved, and the success criterion needs strengthening.", BODY)

P("7.3 Predictions recorded before the runs", H2)
rows = [["#", "prediction", "outcome"],
        ["P8", "null controls sit at chance", "confirmed"],
        ["P12", "closed form matches measurement", "confirmed, 4x10-4"],
        ["P9", "rotation moves AUC toward chance", "FAILED"],
        ["P10", "label-noise arm crosses chance", "FAILED"],
        ["P1", "arm ratio equals tr(inv Lambda)/D", "mixed: 4.58 vs 4.55 on "
         "rot_flat, off 3.6x on rot_mid"],
        ["P11", "input-noise drop is all masking", "true below unit variance, "
         "false above"]]
S.append(table(rows, [1.2*cm, 6.4*cm, 7.2*cm], align_right_from=2))
GAP(3)
P("Recording the failures is the point of having written the predictions down "
  "first. Two of six were wrong, and one of those &mdash; that equalising the "
  "spectra would restore chance &mdash; is the most consequential result here.", CAP)

P("7.4 Files", H2)
P("policy.py (Stage 2 policies and estimators) &middot; theory.py (closed forms, "
  "now including the Bernoulli arm) &middot; corrupt.py (corruption families) "
  "&middot; stage2_optimise.py (Stage 2 driver) &middot; stage2_poc.py (the "
  "synthetic demonstration reported in section 6) &middot; verify_bern.py "
  "(Monte-Carlo validation) &middot; check_null.py (the corrected null test) "
  "&middot; PREDICTIONS.md (pre-registration)", BODY)

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="In-context unlearning: Stage 1 and Stage 2")
doc.build(S)
print("written", OUT, OUT.stat().st_size, "bytes")
