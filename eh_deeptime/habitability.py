"""A guild-mixture Bayesian-logistic habitability metric (illustrative).

For each microbial-metabolism GUILD (thermophile, psychrophile, acidophile,
halophile/xerophile) we fit a logistic growth-probability model in three
environmental features -- temperature (degC), pH and water activity (a_w) --
with pairwise interactions, by MAP (penalised) IRLS / Newton iteration under
Gaussian priors. The habitability of an environment is the across-guild
*maximum* growth probability: an environment is habitable if SOME guild can
grow there (Bayesian-logistic niche mixture).

WHAT THE DATA ARE -- READ THIS. There is NO real extremophile growth database
bundled, fetched, or referenced here. ``make_synthetic_data`` draws SYNTHETIC
environmental points and labels them by whether they fall inside a guild's
published cardinal-tolerance box (T, pH, a_w), softened by a logistic boundary
and Bernoulli label noise. The tolerance boxes (the GUILDS table) are taken
from published microbial cardinal ranges; the (x, y) growth pairs are
fabricated from those envelopes, NOT measured. Every metric below is therefore
a sanity/plausibility check on the *method* (can a penalised logistic recover a
known box from noisy synthetic points?), NOT a calibration to, or validation
against, any real organism.

This is an ILLUSTRATION of the proposed habitability-metric machinery, not a
calibrated or validated model.

PRODUCTION SWAP: a real metric would (a) ingest a curated extremophile
growth/no-growth database (BacDive / DSMZ / literature compilations) instead of
synthetic box labels, and (b) fit a full hierarchical Bayesian model -- guilds
as a partially-pooled hierarchy, posterior over all coefficients -- via HMC/NUTS
(PyMC or Stan) rather than the single-point MAP/IRLS used here for a numpy-only,
offline illustration. Probabilities here are MAP plug-ins, not posterior means.
"""
from __future__ import annotations

import numpy as np

# --- feature order (column order of every X passed around) --------------------
FEATURES = ("T_C", "pH", "a_w")   # temperature (degC), pH, water activity (-)

# ordered guild-name tuple (matches the guild_id integer indexing); filled in
# right after GUILDS is defined below.
GUILD_NAMES = ()

# --- guild cardinal-tolerance boxes ------------------------------------------
# Published microbial cardinal ranges (illustrative envelopes, NOT measured
# growth data). Each guild is a box (lo, hi) per feature; a point inside the box
# is "growth-permissive" for that guild before noise/soft-boundary are applied.
GUILDS = {
    "thermophile":        {"T_C": (45.0, 122.0), "pH": (2.0, 9.0),  "a_w": (0.85, 1.00)},
    "psychrophile":       {"T_C": (-15.0, 20.0), "pH": (5.0, 9.5),  "a_w": (0.80, 1.00)},
    "acidophile":         {"T_C": (10.0, 60.0),  "pH": (0.5, 4.5),  "a_w": (0.90, 1.00)},
    "halophile/xerophile":{"T_C": (5.0, 55.0),   "pH": (6.0, 10.0), "a_w": (0.60, 0.95)},
}
GUILD_NAMES = tuple(GUILDS.keys())

# --- synthetic-sampling envelope ---------------------------------------------
# Plausible global ranges over which environments are drawn (wider than any one
# box, so that "far outside all boxes" points exist). Illustrative.
SAMPLE_RANGE = {"T_C": (-25.0, 130.0), "pH": (-0.5, 11.0), "a_w": (0.50, 1.00)}

# --- soft-boundary / label-noise knobs ---------------------------------------
BOUNDARY_SHARP = 16.0   # logistic sharpness of the soft tolerance-box boundary
LABEL_NOISE = 0.05      # Bernoulli flip probability on the synthetic labels

# --- prior scales (MAP regularisation; standardised-feature space) -----------
PRIOR_SD_MAIN = 5.0     # b ~ N(0, 5) on intercept + main effects
PRIOR_SD_INTERACT = 2.0  # g ~ N(0, 2) on pairwise interactions


def _design(X):
    """Build the standardised design matrix [1, x_i, x_i*x_j] and return helpers.

    Returns (Phi, mu, sd, prior_sd) where Phi has columns
    [intercept, T, pH, a_w, T*pH, T*a_w, pH*a_w] in standardised features.
    ``mu``/``sd`` standardise raw features; ``prior_sd`` is the per-column prior.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)   # guard degenerate columns
    return _design_with(X, mu, sd)


def _design_with(X, mu, sd):
    """Design matrix using *given* standardisation (so test points match train)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    Z = (X - mu) / sd
    z0, z1, z2 = Z[:, 0], Z[:, 1], Z[:, 2]
    Phi = np.column_stack([
        np.ones(len(Z)),            # intercept
        z0, z1, z2,                 # main effects (T, pH, a_w)
        z0 * z0, z1 * z1, z2 * z2,  # quadratic terms (needed to bound a niche box)
        z0 * z1, z0 * z2, z1 * z2,  # pairwise interactions
    ])
    # quadratic terms share the interaction prior (both are second-order shape
    # terms; the box-bounding curvature lives here)
    prior_sd = np.array([
        PRIOR_SD_MAIN, PRIOR_SD_MAIN, PRIOR_SD_MAIN, PRIOR_SD_MAIN,
        PRIOR_SD_INTERACT, PRIOR_SD_INTERACT, PRIOR_SD_INTERACT,
        PRIOR_SD_INTERACT, PRIOR_SD_INTERACT, PRIOR_SD_INTERACT,
    ])
    return Phi, mu, sd, prior_sd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _irls(Phi, y, prior_sd, n_iter=100, tol=1e-8):
    """MAP logistic regression by penalised IRLS / Newton.

    Maximises log-likelihood + Gaussian log-prior (ridge in standardised space
    with per-column scale ``prior_sd``). Returns the MAP coefficient vector.
    """
    n, p = Phi.shape
    beta = np.zeros(p)
    lam = 1.0 / np.asarray(prior_sd) ** 2   # prior precision per coefficient
    Lam = np.diag(lam)
    prev = np.inf
    for _ in range(n_iter):
        eta = Phi @ beta
        mu = _sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-9, None)     # IRLS weights
        # Newton step on penalised objective: H = Phi' W Phi + Lam ; g = Phi'(y-mu) - Lam beta
        WPhi = Phi * w[:, None]
        H = Phi.T @ WPhi + Lam
        g = Phi.T @ (y - mu) - lam * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        nm = float(np.max(np.abs(step)))
        if abs(prev - nm) < tol or nm < tol:
            prev = nm
            break
        prev = nm
    return beta


def _laplace_cov(Phi, beta, prior_sd):
    """Laplace posterior covariance = inverse penalised Hessian at the MAP.

    H = Phi' W Phi + Lam, with W = diag(p(1-p)) at the MAP and Lam the prior
    precision; cov = H^{-1}. This is the standard Laplace (Gaussian) approximation
    to the Bayesian logistic posterior -- the honest stand-in for full HMC, which is
    unavailable here (no PyMC/Stan). Predictive uncertainty on P(growth) is
    propagated from this covariance in :func:`p_hab_ci`.
    """
    lam = 1.0 / np.asarray(prior_sd) ** 2
    p = _sigmoid(Phi @ beta)
    w = np.clip(p * (1.0 - p), 1e-9, None)
    H = Phi.T @ (Phi * w[:, None]) + np.diag(lam)
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(H)


def make_synthetic_data(rng, n=4000):
    """Draw SYNTHETIC (X, y, guild_id) growth pseudo-data from the GUILDS boxes.

    For each sample: pick a guild uniformly, draw an environmental point either
    inside that guild's tolerance box (~60%) or from the wider SAMPLE_RANGE
    (~40%, so the negatives and out-of-box points exist), then assign a growth
    label y in {0,1}. The label uses a SOFT box-membership probability
    (product of per-feature logistic gates, sharpness BOUNDARY_SHARP) followed by
    a Bernoulli draw with LABEL_NOISE flips. ``guild_id`` records the guild whose
    box was used to draw/label the point.

    This is fabricated from PUBLISHED TOLERANCE ENVELOPES -- it is NOT a real
    extremophile growth database (none is bundled or fetched).

    Returns
    -------
    X : (n, 3) float    columns in FEATURES order (T_C, pH, a_w)
    y : (n,) int        growth label in {0, 1}
    guild_id : (n,) int index into GUILD_NAMES of the guild used per sample
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    names = list(GUILDS.keys())
    nf = len(FEATURES)

    gid = rng.integers(0, len(names), size=n)
    X = np.empty((n, nf), dtype=float)
    inside_draw = rng.random(n) < 0.60   # fraction sampled inside the chosen box

    for j, f in enumerate(FEATURES):
        col = np.empty(n)
        # global-range draw for everyone (used where not inside_draw)
        rlo, rhi = SAMPLE_RANGE[f]
        col[:] = rng.uniform(rlo, rhi, size=n)
        # overwrite the in-box draws with a within-box uniform for their guild
        for g, name in enumerate(names):
            blo, bhi = GUILDS[name][f]
            sel = inside_draw & (gid == g)
            if np.any(sel):
                col[sel] = rng.uniform(blo, bhi, size=int(sel.sum()))
        X[:, j] = col

    # Soft box-membership probability of EVERY point in EVERY guild's box
    # (product of per-feature logistic gates). The true growth label is the
    # across-guild MAXIMUM membership -- a point is "habitable" if SOME guild can
    # grow there -- which is exactly what the mixture metric (max over guilds)
    # estimates. Per-guild training labels (used by fit_all / CV) are derived
    # from each guild's own membership column.
    memb = _membership_matrix(X)            # (n, n_guilds) in [0, 1]
    p_any = memb.max(axis=1)                 # true "any guild grows" probability
    p_grow = (1.0 - LABEL_NOISE) * p_any + LABEL_NOISE * (1.0 - p_any)
    y = (rng.random(n) < p_grow).astype(int)
    return X, y, gid


def _membership_matrix(X):
    """Soft tolerance-box membership of each point in each guild (n, n_guilds)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n = len(X)
    M = np.ones((n, len(GUILDS)))
    for g, name in enumerate(GUILDS):
        for j, f in enumerate(FEATURES):
            blo, bhi = GUILDS[name][f]
            span = bhi - blo
            scale = BOUNDARY_SHARP / span
            gate = _sigmoid(scale * (X[:, j] - blo)) * _sigmoid(scale * (bhi - X[:, j]))
            M[:, g] *= gate
    return M


def _guild_label(X, guild_index, rng):
    """Synthetic per-guild growth labels for training one guild's logistic.

    Label is a Bernoulli draw from that guild's OWN soft box-membership (with
    LABEL_NOISE flips), so each guild's model learns its own niche cleanly.
    """
    memb = _membership_matrix(X)[:, guild_index]
    p_grow = (1.0 - LABEL_NOISE) * memb + LABEL_NOISE * (1.0 - memb)
    return (rng.random(len(memb)) < p_grow).astype(int)


def _guild_training_set(rng, guild_index, n, in_box_frac=0.62):
    """Draw a SYNTHETIC, guild-enriched training cloud for one guild.

    Half the points (``in_box_frac``) are drawn uniformly inside the target
    guild's tolerance box (clean positives), half from the wider SAMPLE_RANGE
    (negatives + boundary cases). Labels are Bernoulli draws from that guild's
    soft membership (with LABEL_NOISE). Enriching positives keeps the one-vs-rest
    niche fit from being swamped by the low global base rate. SYNTHETIC, from
    published tolerance envelopes -- not measured growth data.
    """
    name = GUILD_NAMES[guild_index]
    nf = len(FEATURES)
    in_box = rng.random(n) < in_box_frac
    X = np.empty((n, nf))
    for j, f in enumerate(FEATURES):
        rlo, rhi = SAMPLE_RANGE[f]
        col = rng.uniform(rlo, rhi, size=n)
        blo, bhi = GUILDS[name][f]
        col[in_box] = rng.uniform(blo, bhi, size=int(in_box.sum()))
        X[:, j] = col
    yg = _guild_label(X, guild_index, rng)
    return X, yg


def _fit_guild(rng, guild_index, n=4000):
    """Fit one guild's MAP-logistic niche classifier (one-vs-rest, quadratic).

    Trains on a guild-enriched synthetic cloud: positives are points inside the
    guild's tolerance box, negatives are environments outside it. The quadratic +
    interaction design lets the logistic carve out the bounded niche. Returns a
    model dict with keys name, beta, mu, sd, prior_sd, n_train, n_pos.
    """
    X, yg = _guild_training_set(rng, guild_index, n)
    Phi, mu, sd, prior_sd = _design(X)
    beta = _irls(Phi, yg.astype(float), prior_sd)
    cov = _laplace_cov(Phi, beta, prior_sd)   # Laplace posterior covariance
    return {"name": GUILD_NAMES[guild_index], "beta": beta, "cov": cov, "mu": mu,
            "sd": sd, "prior_sd": prior_sd, "n_train": int(n), "n_pos": int(yg.sum())}


def fit_one(rng, guild_index, n=4000):
    """Fit the single-guild MAP-logistic model for one guild (on synthetic data)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    return _fit_guild(rng, guild_index, n=n)


def fit_all(rng=None):
    """Fit a per-guild MAP-logistic model for every guild.

    Each guild is a one-vs-rest niche classifier (positives = inside that guild's
    box) on its own guild-enriched synthetic cloud. Returns a list of model dicts
    in GUILD_NAMES order, each with keys name, beta, mu, sd, prior_sd, n_train.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    return [_fit_guild(rng, g) for g in range(len(GUILD_NAMES))]


def p_hab(X, model):
    """Single-guild growth probability P(growth | x) under one fitted model."""
    Phi, _, _, _ = _design_with(X, model["mu"], model["sd"])
    return _sigmoid(Phi @ model["beta"])


def p_hab_mixture(X, models):
    """Mixture habitability: across-guild MAX growth probability at each x.

    An environment is habitable if at least one guild can grow there, so the
    metric is the maximum single-guild probability over the guild ensemble.
    """
    ps = np.stack([p_hab(X, m) for m in models], axis=0)   # (n_guilds, n)
    return ps.max(axis=0)


def p_hab_ci(X, model, n_sigma=1.64):
    """Single-guild P(growth) with a Laplace-posterior credible interval.

    Returns (central, lo, hi). The logit is approximately Gaussian with mean
    Phi@beta and variance Phi cov Phi' (Laplace). The central value is the plug-in
    prediction sigmoid(mu) -- identical to :func:`p_hab` -- and the band maps the
    logit credible interval mu +- n_sigma*sd through the sigmoid (n_sigma=1.64 ->
    ~90%), so it always nests: lo <= central <= hi. If the model has no Laplace
    'cov' (older fit) the band collapses to the point. This is the honest predictive
    uncertainty the previous MAP plug-in lacked, given no PyMC/Stan/HMC.
    """
    Phi, _, _, _ = _design_with(X, model["mu"], model["sd"])
    mu_logit = Phi @ model["beta"]
    central = _sigmoid(mu_logit)
    if "cov" not in model:
        return central, central, central
    var = np.clip(np.einsum("ij,jk,ik->i", Phi, model["cov"], Phi), 0.0, None)
    sd = np.sqrt(var)
    return (central,
            _sigmoid(mu_logit - n_sigma * sd),
            _sigmoid(mu_logit + n_sigma * sd))


# --- diagnostics --------------------------------------------------------------
def _log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _auc(y, p):
    """Area under ROC by the rank (Mann-Whitney U) estimator."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _calibration_slope(y, p, eps=1e-12):
    """Slope of a logistic recalibration y ~ sigma(a + s * logit(p)).

    A well-calibrated model gives slope s ~ 1. Estimated by a 1-feature MAP
    logistic (weak prior) on the held-out logits.
    """
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    Phi = np.column_stack([np.ones_like(logit), logit])
    prior_sd = np.array([10.0, 10.0])    # weak prior, mostly likelihood-driven
    beta = _irls(Phi, y.astype(float), prior_sd)
    return float(beta[1])


def grouped_cross_validate(rng=None, k=5):
    """k-fold cross-validation of the mixture metric, folds grouped BY GUILD.

    Within each guild the points are split into k folds; for each fold we train
    every guild on its remaining points and score the held-out points of all
    guilds with the across-guild mixture probability. Grouping by guild keeps the
    same guild's points from leaking train->test in a way that would flatter the
    metric. All metrics below are sanity/plausibility checks on the method, NOT a
    validation against real organisms.

    Returns
    -------
    dict with keys: log_loss, brier, calibration_slope, auc, baseline_brier
                    (baseline_brier = Brier of the constant base-rate predictor).
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    X, y, gid = make_synthetic_data(rng, n=4000)
    n = len(y)

    # per-guild fold assignment (group by guild): shuffle each guild's indices
    # and tile fold ids, so every fold holds a slice of every guild.
    fold = np.empty(n, dtype=int)
    for g in range(len(GUILD_NAMES)):
        idx = np.where(gid == g)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k

    p_oof = np.empty(n, dtype=float)   # out-of-fold mixture predictions
    for f in range(k):
        test = fold == f
        # train each guild's one-vs-rest niche on a fresh guild-enriched
        # synthetic cloud (independent of the held-out test rows -> no leakage),
        # then score the held-out points with the across-guild mixture.
        models = [_fit_guild(rng, g, n=3000) for g in range(len(GUILD_NAMES))]
        p_oof[test] = p_hab_mixture(X[test], models)

    base_rate = float(y.mean())
    return {
        "log_loss": _log_loss(y, p_oof),
        "brier": _brier(y, p_oof),
        "calibration_slope": _calibration_slope(y, p_oof),
        "auc": _auc(y, p_oof),
        "baseline_brier": _brier(y, np.full(n, base_rate)),
    }


def box_centre(guild_name):
    """Return the (T_C, pH, a_w) midpoint of a guild's tolerance box (helper)."""
    g = GUILDS[guild_name]
    return np.array([(g[f][0] + g[f][1]) / 2.0 for f in FEATURES])


def summarise(rng=None):
    """One-line diagnostic bundle for the fitted mixture (illustrative)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    models = fit_all(rng)
    centres = np.array([box_centre(n) for n in GUILD_NAMES])
    p_at_centres = p_hab_mixture(centres, models)
    cv = grouped_cross_validate(rng)
    return {"guilds": GUILD_NAMES,
            "p_hab_at_box_centres": p_at_centres,
            "cv": cv}
