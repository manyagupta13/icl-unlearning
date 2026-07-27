"""
Independent NumPy check of the closed-form claims in NOTES.md section 0.

Reimplements the forward pass and the C1/C2 corruptions from scratch (i.e.
NOT by importing the repo), then compares Monte-Carlo moments of yhat against
the analytic expressions. If these agree, the claim that C1 carries an O(sigma^2)
mean shift and C2 does not is correct.
"""
import numpy as np

rng = np.random.default_rng(0)

D, N = 4, 31
n_f = 11                      # forget-group context tokens, placed last
f = slice(N - n_f, N)

# --- a probe point: context tokens + query, and a read-out matrix ------------
x = rng.normal(size=(N + 1, D))
beta = rng.normal(size=D)
y_true = x @ beta                                  # labels for all N+1
M = rng.normal(size=(D + 1, D + 1)) * 0.3


def forward(x, y):
    """yhat = (1/(N+1)) t_q^T M u,  u = sum_i t_i y_i, query label slot zeroed."""
    ylab = y.copy()
    ylab[-1] = 0.0
    X = np.concatenate([x, ylab[:, None]], axis=1)   # [N+1, D+1] tokens
    u = X.T @ ylab                                   # [D+1]
    tq = X[-1]
    return (tq @ M @ u) / (N + 1)


# c = M^T t_q, split into c_x (R^D) and c_y (scalar)
ylab0 = y_true.copy(); ylab0[-1] = 0.0
X0 = np.concatenate([x, ylab0[:, None]], axis=1)
tq = X0[-1]
c = M.T @ tq
c_x, c_y = c[:D], c[D]

yhat_clean = forward(x, y_true)

TRIALS = 400_000
print(f"{'sigma^2':>9} | {'C1 dMean emp':>13} {'C1 dMean thy':>13} | "
      f"{'C1 Var emp':>11} {'C1 Var thy':>11} | {'C2 dMean emp':>13} "
      f"{'C2 Var emp':>11} {'C2 Var thy':>11}")
print("-" * 108)

for s2 in (0.01, 0.1, 1.0, 4.0):
    s = np.sqrt(s2)

    # ---- C1: label noise on forget tokens --------------------------------
    eps = rng.normal(scale=s, size=(TRIALS, n_f))
    yc = np.tile(y_true, (TRIALS, 1))
    yc[:, f] += eps
    ylab = yc.copy(); ylab[:, -1] = 0.0
    Xc = np.concatenate([np.tile(x, (TRIALS, 1, 1)), ylab[:, :, None]], axis=2)
    u = np.einsum("tnd,tn->td", Xc, ylab)
    yh1 = np.einsum("d,de,te->t", tq, M, u) / (N + 1)

    d_mean_emp = yh1.mean() - yhat_clean
    d_mean_thy = c_y * n_f * s2 / (N + 1)

    var_emp = yh1.var()
    lin = c_x @ x[f].T + 2 * c_y * y_true[f]          # [n_f]
    var_thy = (s2 * (lin ** 2).sum() + 2 * c_y ** 2 * n_f * s2 ** 2) / (N + 1) ** 2

    # ---- C2: input noise on forget tokens --------------------------------
    eta = rng.normal(scale=s, size=(TRIALS, n_f, D))
    xc = np.tile(x, (TRIALS, 1, 1))
    xc[:, f, :] += eta
    ylab2 = np.tile(y_true, (TRIALS, 1)); ylab2[:, -1] = 0.0
    Xc2 = np.concatenate([xc, ylab2[:, :, None]], axis=2)
    u2 = np.einsum("tnd,tn->td", Xc2, ylab2)
    tq2 = Xc2[:, -1, :]                               # query token unchanged
    yh2 = np.einsum("td,de,te->t", tq2, M, u2) / (N + 1)

    d2_emp = yh2.mean() - yhat_clean
    var2_emp = yh2.var()
    var2_thy = s2 * (c_x @ c_x) * (y_true[f] ** 2).sum() / (N + 1) ** 2

    print(f"{s2:9.3g} | {d_mean_emp:13.5f} {d_mean_thy:13.5f} | "
          f"{var_emp:11.5f} {var_thy:11.5f} | {d2_emp:13.5f} "
          f"{var2_emp:11.5f} {var2_thy:11.5f}")

print()
print("PASS criteria: C1 dMean emp ~= thy and grows linearly in sigma^2;")
print("               C2 dMean emp ~= 0 at all sigma^2.")
