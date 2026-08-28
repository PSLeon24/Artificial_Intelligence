# -*- coding: utf-8 -*-
"""Figures for the AI notes: every one is a real experiment, not a diagram.

Thirteen panels. The clustering ones run the algorithms, the classifiers are
trained and scored on held-out data, and the claims in the captions are numbers
those runs produced. Where a method fails, the figure shows it failing rather
than showing the case where it works.

    python make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "figures"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#999999"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# numpy 2 removed np.trapz in favour of np.trapezoid
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def blobs(rng, n=300):
    c = np.array([[-2.2, -1.6], [2.4, -1.2], [0.2, 2.6]])
    y = rng.integers(0, 3, n)
    return c[y] + rng.normal(0, 0.85, (n, 2)), y


def kmeans(X, k, C0, iters=12):
    C = C0.copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C, ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


def fig_kmeans():
    rng = np.random.default_rng(0)
    X, _ = blobs(rng)
    C = np.array([[-3.0, 2.6], [-2.6, 2.0], [-2.2, 2.4]])
    snaps, inertias = [], []
    for _ in range(7):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        inertias.append(d.min(1).sum())
        snaps.append((C.copy(), a.copy()))
        for k in range(3):
            if (a == k).any():
                C[k] = X[a == k].mean(0)
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9))
    for ax, i in zip(axes[:4], [0, 1, 2, 6]):
        Ci, ai = snaps[i]
        for k, col in enumerate((BLUE, ORANGE, GREEN)):
            ax.scatter(*X[ai == k].T, s=9, color=col, alpha=0.65)
            ax.plot(*Ci[k], "X", color=col, ms=13, mec="black", mew=1.1)
        ax.set_title("iteration %d\ninertia %.0f" % (i, inertias[i]), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[4].plot(inertias, "o-", color=BLUE)
    axes[4].set_title("Inertia never increases\n(the same guarantee EM has)", fontsize=9)
    axes[4].set_xlabel("iteration")
    fig.suptitle("k-means recovers three groups from a start with all centroids "
                 "in one corner", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "kmeans.png"))
    plt.close(fig)


def fig_clustering_compare():
    """Where k-means fails and DBSCAN does not, and the reverse."""
    rng = np.random.default_rng(1)
    t = rng.uniform(0, np.pi, 200)
    moon1 = np.stack([np.cos(t) * 2, np.sin(t) * 2], 1) + rng.normal(0, 0.16, (200, 2))
    moon2 = np.stack([np.cos(t) * 2 + 2, -np.sin(t) * 2 + 0.8], 1) \
        + rng.normal(0, 0.16, (200, 2))
    X = np.vstack([moon1, moon2])

    def dbscan(X, eps, minpts):
        n = len(X)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        nb = [np.where(D[i] <= eps)[0] for i in range(n)]
        core = np.array([len(nb[i]) >= minpts for i in range(n)])
        lab = np.full(n, -1)
        c = 0
        for i in range(n):
            if not core[i] or lab[i] != -1:
                continue
            stack, lab[i] = [i], c
            while stack:
                j = stack.pop()
                for q in nb[j]:
                    if lab[q] == -1:
                        lab[q] = c
                        if core[q]:
                            stack.append(q)
            c += 1
        return lab, c

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    C0 = X[rng.choice(len(X), 2, replace=False)]
    _, a = kmeans(X, 2, C0)
    for k, col in enumerate((BLUE, ORANGE)):
        axes[0].scatter(*X[a == k].T, s=9, color=col)
    axes[0].set_title("k-means, k = 2\nsplits the moons down the middle", fontsize=9)
    lab, nc = dbscan(X, 0.42, 5)
    for k in range(nc):
        axes[1].scatter(*X[lab == k].T, s=9, color=[BLUE, ORANGE, GREEN, PURPLE][k % 4])
    axes[1].scatter(*X[lab == -1].T, s=14, color=GREY, marker="x")
    axes[1].set_title("DBSCAN eps=0.42, minPts=5\n%d clusters, %d points marked noise"
                      % (nc, int((lab == -1).sum())), fontsize=9)
    Xb, _ = blobs(rng, 300)
    lab2, nc2 = dbscan(Xb, 0.42, 5)
    for k in range(nc2):
        axes[2].scatter(*Xb[lab2 == k].T, s=9, color=[BLUE, ORANGE, GREEN, PURPLE][k % 4])
    axes[2].scatter(*Xb[lab2 == -1].T, s=14, color=GREY, marker="x")
    axes[2].set_title("Same DBSCAN settings on blobs\n%d clusters, %d noise — eps does "
                      "not transfer" % (nc2, int((lab2 == -1).sum())), fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("k-means assumes round clusters; DBSCAN does not, but its eps has to "
                 "be retuned per dataset", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "clustering_compare.png"))
    plt.close(fig)


def fig_dendrogram():
    """Agglomerative clustering, built bottom-up and drawn as the merge tree."""
    rng = np.random.default_rng(4)
    X, _ = blobs(rng, 18)
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    clusters = {i: [i] for i in range(n)}
    pos = {i: (i, 0.0) for i in range(n)}
    order = list(range(n))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    a1.scatter(*X.T, s=26, color=BLUE)
    for i in range(n):
        a1.annotate(str(i), X[i] + 0.12, fontsize=7, color="#444444")
    a1.set_title("18 points", fontsize=10)
    a1.set_xticks([]); a1.set_yticks([])
    active = list(range(n))
    nxt = n
    while len(active) > 1:
        best, pair = np.inf, None
        for i in active:
            for j in active:
                if i < j:
                    d = max(D[a][b] for a in clusters[i] for b in clusters[j])  # complete link
                    if d < best:
                        best, pair = d, (i, j)
        i, j = pair
        x = 0.5 * (pos[i][0] + pos[j][0])
        a2.plot([pos[i][0], pos[i][0], pos[j][0], pos[j][0]],
                [pos[i][1], best, best, pos[j][1]], color=BLUE, lw=1.2)
        clusters[nxt] = clusters[i] + clusters[j]
        pos[nxt] = (x, best)
        active = [k for k in active if k not in pair] + [nxt]
        nxt += 1
    a2.set_xticks([])
    a2.set_ylabel("merge distance (complete linkage)")
    a2.set_title("Dendrogram: cutting at a height\nchooses the number of clusters",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dendrogram.png"))
    plt.close(fig)


def fig_knn():
    rng = np.random.default_rng(1)
    n = 260
    t = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, 1, n)
    X = np.stack([r * np.cos(t) * 3, r * np.sin(t) * 3], 1)
    y = ((X[:, 1] > 0.55 * X[:, 0] ** 2 - 1.6).astype(int)
         ^ (rng.random(n) < 0.10).astype(int))
    tr, te = slice(0, 180), slice(180, n)

    def knn(Xtr, ytr, Q, k):
        d = ((Q[:, None, :] - Xtr[None]) ** 2).sum(-1)
        idx = np.argsort(d, 1)[:, :k]
        return (ytr[idx].mean(1) > 0.5).astype(int)

    g = np.linspace(-3.4, 3.4, 220)
    GX, GY = np.meshgrid(g, g)
    Q = np.stack([GX.ravel(), GY.ravel()], 1)
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.2))
    for ax, k in zip(axes[:3], (1, 15, 91)):
        Z = knn(X[tr], y[tr], Q, k).reshape(GX.shape)
        ax.contourf(GX, GY, Z, levels=1, colors=["#dfe9f6", "#fbe4d8"])
        ax.scatter(*X[tr][y[tr] == 0].T, s=12, color=BLUE)
        ax.scatter(*X[tr][y[tr] == 1].T, s=12, color=ORANGE)
        acc = (knn(X[tr], y[tr], X[te], k) == y[te]).mean()
        ax.set_title("k = %d   test acc %.3f" % (k, acc), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    kk = list(range(1, 100, 2))
    tr_e = [1 - (knn(X[tr], y[tr], X[tr], k) == y[tr]).mean() for k in kk]
    te_e = [1 - (knn(X[tr], y[tr], X[te], k) == y[te]).mean() for k in kk]
    axes[3].plot(kk, tr_e, color=GREY, lw=1.6, label="train error")
    axes[3].plot(kk, te_e, color=BLUE, lw=2, label="test error")
    axes[3].axvline(kk[int(np.argmin(te_e))], color=ORANGE, ls="--", lw=1.2)
    axes[3].set_xlabel("k"); axes[3].set_ylabel("error")
    axes[3].set_title("k = 1 fits the noise (train error 0);\nbest k = %d"
                      % kk[int(np.argmin(te_e))], fontsize=9)
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "knn.png"))
    plt.close(fig)


def fig_naivebayes():
    """The independence assumption, and what it costs when it is false."""
    rng = np.random.default_rng(6)
    n = 400
    m0, m1 = np.array([-1.0, -0.8]), np.array([1.0, 0.9])
    S = np.array([[1.0, 0.86], [0.86, 1.0]])       # strongly correlated features
    L = np.linalg.cholesky(S)
    X = np.vstack([(L @ rng.normal(size=(2, n))).T + m0,
                   (L @ rng.normal(size=(2, n))).T + m1])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    g = np.linspace(-5, 5, 260)
    GX, GY = np.meshgrid(g, g)

    def gauss(x, m, cov):
        d = x - m
        inv = np.linalg.inv(cov)
        return np.exp(-0.5 * np.einsum("...i,ij,...j", d, inv, d)) \
            / np.sqrt(np.linalg.det(cov))

    Q = np.stack([GX.ravel(), GY.ravel()], 1)
    diag = np.diag(np.diag(S))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    for ax, cov, name in ((axes[0], diag, "Naive Bayes\n(features assumed independent)"),
                          (axes[1], S, "Full covariance\n(correlation modelled)")):
        Z = (gauss(Q, m1, cov) > gauss(Q, m0, cov)).reshape(GX.shape)
        acc = ((gauss(X, m1, cov) > gauss(X, m0, cov)).astype(int) == y).mean()
        ax.contourf(GX, GY, Z, levels=1, colors=["#dfe9f6", "#fbe4d8"])
        ax.scatter(*X[y == 0].T, s=6, color=BLUE, alpha=0.6)
        ax.scatter(*X[y == 1].T, s=6, color=ORANGE, alpha=0.6)
        ax.set_title("%s\ntrain acc %.3f" % (name, acc), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    cors = np.linspace(0, 0.95, 20)
    accs = []
    for c in cors:
        Sc = np.array([[1.0, c], [c, 1.0]])
        Lc = np.linalg.cholesky(Sc)
        Xi = np.vstack([(Lc @ rng.normal(size=(2, n))).T + m0,
                        (Lc @ rng.normal(size=(2, n))).T + m1])
        dg = np.diag(np.diag(Sc))
        accs.append(((gauss(Xi, m1, dg) > gauss(Xi, m0, dg)).astype(int) == y).mean())
    axes[2].plot(cors, accs, "o-", color=BLUE, ms=3)
    axes[2].set_xlabel("true correlation between features")
    axes[2].set_ylabel("Naive Bayes accuracy")
    axes[2].set_title("The assumption degrades gracefully:\nstill useful even when "
                      "plainly false", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "naivebayes.png"))
    plt.close(fig)


def fig_logistic():
    """Odds, logit, sigmoid, and a fitted boundary."""
    p = np.linspace(0.001, 0.999, 500)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
    axes[0].plot(p, p / (1 - p), color=BLUE, lw=2)
    axes[0].set_ylim(0, 20)
    axes[0].set_xlabel("p"); axes[0].set_ylabel("odds = p/(1−p)")
    axes[0].set_title("Odds: 0 to ∞, and it explodes\nas p approaches 1", fontsize=9)
    axes[1].plot(p, np.log(p / (1 - p)), color=ORANGE, lw=2)
    axes[1].axhline(0, color=GREY, ls=":", lw=1)
    axes[1].axvline(0.5, color=GREY, ls=":", lw=1)
    axes[1].set_xlabel("p"); axes[1].set_ylabel("logit(p)")
    axes[1].set_title("Logit: −∞ to ∞, symmetric,\n0 exactly at p = 0.5", fontsize=9)

    rng = np.random.default_rng(8)
    n = 200
    X = rng.normal(0, 1.6, (n, 2))
    w_true = np.array([1.6, -1.1])
    y = (1 / (1 + np.exp(-(X @ w_true + 0.4))) > rng.random(n)).astype(float)
    w, b = np.zeros(2), 0.0
    for _ in range(4000):
        z = X @ w + b
        pr = 1 / (1 + np.exp(-z))
        gw = X.T @ (pr - y) / n
        gb = (pr - y).mean()
        w -= 0.6 * gw; b -= 0.6 * gb
    g = np.linspace(-5, 5, 200)
    GX, GY = np.meshgrid(g, g)
    Z = 1 / (1 + np.exp(-(np.stack([GX.ravel(), GY.ravel()], 1) @ w + b)))
    cs = axes[2].contourf(GX, GY, Z.reshape(GX.shape), levels=20, cmap="RdBu_r", alpha=0.75)
    axes[2].contour(GX, GY, Z.reshape(GX.shape), levels=[0.5], colors="black",
                    linewidths=1.6)
    axes[2].scatter(*X[y == 0].T, s=10, color=BLUE)
    axes[2].scatter(*X[y == 1].T, s=10, color=ORANGE)
    acc = (((1 / (1 + np.exp(-(X @ w + b)))) > 0.5).astype(float) == y).mean()
    axes[2].set_title("Fitted by gradient descent on the\nlog-loss, train acc %.3f" % acc,
                      fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(cs, ax=axes[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "logistic.png"))
    plt.close(fig)


def fig_perceptron():
    """Perceptron against Adaline on the same separable data."""
    rng = np.random.default_rng(11)
    n = 120
    X = np.vstack([rng.normal([-1.5, -1.0], 0.8, (n, 2)),
                   rng.normal([1.6, 1.2], 0.8, (n, 2))])
    y = np.r_[-np.ones(n), np.ones(n)]
    Xb = np.c_[X, np.ones(len(X))]

    def perceptron(epochs=25, lr=0.02):
        w = np.zeros(3); errs = []
        for _ in range(epochs):
            e = 0
            for i in rng.permutation(len(Xb)):
                if y[i] * (Xb[i] @ w) <= 0:
                    w += lr * y[i] * Xb[i]; e += 1
            errs.append(e)
        return w, errs

    def adaline(epochs=25, lr=0.002):
        w = np.zeros(3); loss = []
        for _ in range(epochs):
            out = Xb @ w
            w -= lr * Xb.T @ (out - y)
            loss.append(((y - out) ** 2).mean())
        return w, loss

    wp, errs = perceptron()
    wa, loss = adaline()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    xs = np.array([-4, 4])
    axes[0].scatter(*X[y < 0].T, s=10, color=BLUE)
    axes[0].scatter(*X[y > 0].T, s=10, color=ORANGE)
    for w, col, name in ((wp, GREEN, "perceptron"), (wa, PURPLE, "Adaline")):
        axes[0].plot(xs, -(w[0] * xs + w[2]) / w[1], color=col, lw=2, label=name)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Both separate the data,\nby different boundaries", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].plot(errs, "o-", color=GREEN, ms=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("misclassifications")
    axes[1].set_title("Perceptron: updates only on mistakes,\nstops when there are none",
                      fontsize=9)
    axes[2].plot(loss, "o-", color=PURPLE, ms=3)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("mean squared error")
    axes[2].set_title("Adaline: minimizes a continuous loss,\nso it keeps improving the "
                      "margin", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "perceptron.png"))
    plt.close(fig)


def fig_activation():
    rng = np.random.default_rng(2)
    n = 800
    X = rng.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(float)
    Xtr, ytr, Xte, yte = X[:600], y[:600], X[600:], y[600:]

    def train(act, hidden=16, epochs=3000, lr=0.08):
        g = np.random.default_rng(0)
        W1 = g.normal(0, 0.8, (2, hidden)); b1 = np.zeros(hidden)
        W2 = g.normal(0, 0.8, (hidden, 1)); b2 = np.zeros(1)
        f = {"relu": lambda z: np.maximum(z, 0), "identity": lambda z: z}[act]
        df = {"relu": lambda z: (z > 0).astype(float),
              "identity": lambda z: np.ones_like(z)}[act]
        for _ in range(epochs):
            z1 = Xtr @ W1 + b1
            h = f(z1)
            o = (h @ W2 + b2).ravel()
            p = 1 / (1 + np.exp(-o))
            d = (p - ytr) / len(ytr)
            gW2 = h.T @ d[:, None]; gb2 = d.sum(keepdims=True)
            dh = d[:, None] @ W2.T * df(z1)
            W1 -= lr * (Xtr.T @ dh); b1 -= lr * dh.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2
        pred = lambda Q: (1 / (1 + np.exp(-((f(Q @ W1 + b1) @ W2 + b2).ravel())))) > 0.5  # noqa: E731
        return pred, (pred(Xte) == yte).mean()

    g = np.linspace(-1, 1, 220)
    GX, GY = np.meshgrid(g, g)
    Q = np.stack([GX.ravel(), GY.ravel()], 1)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, act, name in ((axes[0], "identity", "identity (linear) activation"),
                          (axes[1], "relu", "ReLU activation")):
        pred, acc = train(act)
        ax.contourf(GX, GY, pred(Q).reshape(GX.shape), levels=1,
                    colors=["#dfe9f6", "#fbe4d8"])
        ax.scatter(*Xte[yte == 0].T, s=10, color=BLUE)
        ax.scatter(*Xte[yte == 1].T, s=10, color=ORANGE)
        ax.set_title("%s\n2-layer net, test acc %.3f" % (name, acc), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    z = np.linspace(-4, 4, 300)
    axes[2].plot(z, 1 / (1 + np.exp(-z)), color=BLUE, lw=1.8, label="sigmoid")
    axes[2].plot(z, np.tanh(z), color=ORANGE, lw=1.8, label="tanh")
    axes[2].plot(z, np.maximum(z, 0), color=GREEN, lw=1.8, label="ReLU")
    axes[2].plot(z, np.exp(-z ** 2 * 0) * (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))),
                 color=PURPLE, lw=1.4, ls="--", label="sigmoid'  (max 0.25)")
    axes[2].axhline(0, color="#dddddd", lw=0.8)
    axes[2].set_ylim(-1.6, 3)
    axes[2].set_title("Sigmoid's derivative never exceeds 0.25 —\n"
                      "the vanishing gradient, in one line", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Stacking linear layers is still a linear model: it cannot separate "
                 "XOR no matter how deep it goes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(os.path.join(OUT, "activation.png"))
    plt.close(fig)


def fig_optimizers():
    """Batch, stochastic and mini-batch descent on the same surface."""
    rng = np.random.default_rng(5)
    n = 400
    X = np.c_[rng.normal(0, 1, n), rng.normal(0, 1, n)]
    w_true = np.array([2.0, -1.0])
    y = X @ w_true + rng.normal(0, 0.6, n)
    loss = lambda w: ((X @ w - y) ** 2).mean()      # noqa: E731

    def run(bs, lr, steps=140):
        w = np.array([-2.5, 2.5]); path = [w.copy()]
        for s in range(steps):
            idx = rng.choice(n, bs, replace=False) if bs < n else np.arange(n)
            g = 2 * X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)
            w = w - lr * g
            path.append(w.copy())
        return np.array(path)

    g1 = np.linspace(-3, 3.5, 200)
    A, B = np.meshgrid(g1, g1)
    Z = np.array([loss(np.array([a, b])) for a, b in zip(A.ravel(), B.ravel())]).reshape(A.shape)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    for ax, (bs, lr, name) in zip(axes, ((n, 0.08, "Batch (all %d)" % n),
                                         (1, 0.02, "Stochastic (1 sample)"),
                                         (32, 0.06, "Mini-batch (32)"))):
        p = run(bs, lr)
        ax.contour(A, B, Z, levels=25, colors="#88888855", linewidths=0.7)
        ax.plot(p[:, 0], p[:, 1], "-", color=BLUE, lw=1.1)
        ax.plot(*p[0], "o", color=GREEN, ms=8)
        ax.plot(*w_true, "*", color=ORANGE, ms=14)
        ax.set_title("%s\nfinal loss %.4f" % (name, loss(p[-1])), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Batch descent is smooth and expensive per step; stochastic is noisy "
                 "and cheap; mini-batch is the compromise everything uses", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "optimizers.png"))
    plt.close(fig)


def fig_regularization():
    """L1 and L2 coefficient paths on correlated features with irrelevant columns."""
    rng = np.random.default_rng(9)
    n, p = 80, 8
    X = rng.normal(0, 1, (n, p))
    X[:, 1] = X[:, 0] * 0.9 + rng.normal(0, 0.3, n)     # correlated pair
    w_true = np.array([2.0, 0.0, -1.5, 0, 0, 0.8, 0, 0])
    y = X @ w_true + rng.normal(0, 0.5, n)
    lams = np.logspace(-3, 1.4, 60)
    ridge, lasso = [], []
    for lam in lams:
        ridge.append(np.linalg.solve(X.T @ X + lam * n * np.eye(p), X.T @ y))
        w = np.zeros(p)
        for _ in range(600):                             # coordinate descent
            for j in range(p):
                r = y - X @ w + X[:, j] * w[j]
                rho = X[:, j] @ r / n
                z = (X[:, j] ** 2).mean()
                w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / z
        lasso.append(w.copy())
    ridge, lasso = np.array(ridge), np.array(lasso)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    for ax, path, name in ((axes[0], ridge, "Ridge (L2)"), (axes[1], lasso, "Lasso (L1)")):
        for j in range(p):
            ax.semilogx(lams, path[:, j],
                        color=BLUE if w_true[j] != 0 else GREY,
                        lw=2 if w_true[j] != 0 else 1.1)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("λ"); ax.set_ylabel("coefficient")
        ax.set_title("%s\nblue = truly non-zero, grey = irrelevant" % name, fontsize=9)
    axes[2].semilogx(lams, (np.abs(ridge) < 1e-6).sum(1), color=BLUE, lw=2, label="ridge")
    axes[2].semilogx(lams, (np.abs(lasso) < 1e-6).sum(1), color=ORANGE, lw=2, label="lasso")
    axes[2].set_xlabel("λ"); axes[2].set_ylabel("coefficients exactly zero")
    axes[2].set_title("Only L1 sets coefficients to exactly 0,\nwhich is what makes it "
                      "a selector", fontsize=9)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regularization.png"))
    plt.close(fig)


def fig_convolution():
    """What a convolution kernel actually computes."""
    rng = np.random.default_rng(0)
    g = np.linspace(-3, 3, 64)
    X, Y = np.meshgrid(g, g)
    img = (np.abs(X) < 1.2).astype(float) * (np.abs(Y) < 1.6) \
        + 0.5 * (X ** 2 + Y ** 2 < 0.6)
    img += rng.normal(0, 0.05, img.shape)
    kernels = {
        "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
        "blur 3x3": np.ones((3, 3)) / 9,
        "Sobel x\n(vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    }

    def conv(a, k):
        out = np.zeros((a.shape[0] - 2, a.shape[1] - 2))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = (a[i:i + 3, j:j + 3] * k).sum()
        return out

    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input 64×64", fontsize=9)
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        o = conv(img, k)
        ax.imshow(o, cmap="gray")
        ax.set_title("%s\noutput %d×%d" % (name, o.shape[0], o.shape[1]), fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("A 3×3 kernel with no padding takes 64×64 to 62×62: "
                 "W₂ = (W₁ − F + 2P)/S + 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(os.path.join(OUT, "convolution.png"))
    plt.close(fig)


def fig_collaborative():
    """User-based against item-based similarity on one ratings matrix."""
    rng = np.random.default_rng(12)
    n_u, n_i, k = 14, 10, 3
    U = rng.random((n_u, k))
    V = rng.random((n_i, k))
    R = np.clip(np.round((U @ V.T) * 4 + 1), 1, 5)
    mask = rng.random(R.shape) < 0.35
    Robs = np.where(mask, R, np.nan)

    def cos_sim(M):
        M0 = np.nan_to_num(M - np.nanmean(M, 1, keepdims=True))
        nrm = np.linalg.norm(M0, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        return (M0 / nrm) @ (M0 / nrm).T

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    im = axes[0].imshow(Robs, cmap="YlGnBu", vmin=1, vmax=5)
    axes[0].set_title("Ratings matrix, %.0f%% observed\n(white = missing)"
                      % (100 * mask.mean()), fontsize=9)
    axes[0].set_xlabel("items"); axes[0].set_ylabel("users")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].imshow(cos_sim(Robs), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("User-user similarity (%d×%d)\nrecompute as users arrive"
                      % (n_u, n_u), fontsize=9)
    axes[2].imshow(cos_sim(Robs.T), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("Item-item similarity (%d×%d)\nsmaller and more stable over time"
                      % (n_i, n_i), fontsize=9)
    for ax in axes[1:]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The two views of the same data: item-item is preferred in practice "
                 "because items are fewer and their similarities drift slowly",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "collaborative.png"))
    plt.close(fig)


def fig_metrics():
    rng = np.random.default_rng(5)
    n_pos, n_neg = 120, 880
    s = np.concatenate([rng.normal(1.4, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    th = np.linspace(s.min(), s.max(), 300)
    prec, rec, tpr, fpr, acc = [], [], [], [], []
    for t in th:
        p = s >= t
        tp = float((p & (y == 1)).sum()); fp = float((p & (y == 0)).sum())
        fn = float((~p & (y == 1)).sum()); tn = float((~p & (y == 0)).sum())
        prec.append(tp / max(tp + fp, 1)); rec.append(tp / max(tp + fn, 1))
        tpr.append(tp / max(tp + fn, 1)); fpr.append(fp / max(fp + tn, 1))
        acc.append((tp + tn) / len(y))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    axes[0].plot(th, prec, color=BLUE, lw=1.8, label="precision")
    axes[0].plot(th, rec, color=ORANGE, lw=1.8, label="recall")
    axes[0].plot(th, acc, color=GREY, lw=1.6, ls="--", label="accuracy")
    axes[0].axhline(1 - n_pos / len(y), color=GREEN, lw=1.2, ls=":",
                    label="always-negative baseline")
    axes[0].set_xlabel("decision threshold")
    axes[0].set_title("One knob, opposite effects", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5)
    order = np.argsort(fpr)
    auc = TRAPZ(np.array(tpr)[order], np.array(fpr)[order])
    axes[1].plot(fpr, tpr, color=BLUE, lw=2)
    axes[1].plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.2)
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC, AUC = %.3f" % auc, fontsize=9)
    axes[2].plot(rec, prec, color=ORANGE, lw=2)
    axes[2].axhline(n_pos / len(y), color=GREY, ls=":", lw=1.2)
    axes[2].set_xlabel("recall"); axes[2].set_ylabel("precision")
    axes[2].set_title("Precision-recall\nbaseline = prevalence = %.2f"
                      % (n_pos / len(y)), fontsize=9)
    fig.suptitle("12%% positives: accuracy stays high by predicting negative, "
                 "so it is the misleading metric here", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(os.path.join(OUT, "metrics.png"))
    plt.close(fig)


def fig_biasvariance():
    rng = np.random.default_rng(3)
    f = lambda x: np.sin(1.6 * x) + 0.35 * x       # noqa: E731
    xtr = np.sort(rng.uniform(-3, 3, 22))
    ytr = f(xtr) + rng.normal(0, 0.32, xtr.size)
    xte = np.sort(rng.uniform(-3, 3, 400))
    yte = f(xte) + rng.normal(0, 0.32, xte.size)
    degs = range(1, 16)
    tr_e = [np.mean((np.polyval(np.polyfit(xtr, ytr, d), xtr) - ytr) ** 2) for d in degs]
    te_e = [np.mean((np.polyval(np.polyfit(xtr, ytr, d), xte) - yte) ** 2) for d in degs]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))
    xs = np.linspace(-3.1, 3.1, 400)
    for ax, d, name in ((axes[0], 1, "degree 1 — underfit"),
                        (axes[1], 5, "degree 5 — about right"),
                        (axes[2], 15, "degree 15 — overfit")):
        c = np.polyfit(xtr, ytr, d)
        ax.plot(xs, f(xs), color=GREY, lw=1.4, ls=":", label="true f")
        ax.plot(xs, np.polyval(c, xs), color=BLUE, lw=2, label="fit")
        ax.scatter(xtr, ytr, s=18, color=ORANGE, zorder=3)
        ax.set_ylim(-3, 3)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    axes[3].semilogy(list(degs), tr_e, color=GREY, lw=1.8, label="train MSE")
    axes[3].semilogy(list(degs), te_e, color=BLUE, lw=2, label="test MSE")
    best = list(degs)[int(np.argmin(te_e))]
    axes[3].axvline(best, color=ORANGE, ls="--", lw=1.2)
    axes[3].set_xlabel("polynomial degree")
    axes[3].set_title("Train error keeps falling,\ntest error turns at degree %d" % best,
                      fontsize=9)
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "biasvariance.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_kmeans, fig_clustering_compare, fig_dendrogram, fig_knn,
               fig_naivebayes, fig_logistic, fig_perceptron, fig_activation,
               fig_optimizers, fig_regularization, fig_convolution,
               fig_collaborative, fig_metrics, fig_biasvariance):
        fn()
    print("wrote", sorted(os.listdir(OUT)))
