"""Test-time class-mix correction (CR-2026-09-28-image-competitive-gaps Item 3). SWITCH, DEFAULT OFF.

Expectation-maximization prior adjustment (Saerens, Latinne and Decaestecker, 2002): the model's site probabilities were
fitted under the training class mix pi_train; on an unlabeled test set with a different mix, iterate
    pi <- mean over all sites of p_adj,   p_adj ∝ p * pi / pi_train
for a fixed number of iterations (deterministic, M3-10). No labels are used. Whether DARPA accepts adapting to the
unlabeled test images is Joseph's decision (default off; forum question parked).
"""
import numpy as np


def em_prior(P, pi_train, iters=20, floor=1e-3):
    """P (N, 4) site probabilities; returns (adjusted P, estimated test prior)."""
    P = np.asarray(P, float); pt = np.asarray(pi_train, float); pi = pt.copy()
    for _ in range(int(iters)):
        A = P * (pi / pt); A /= A.sum(1, keepdims=True)
        pi = np.clip(A.mean(0), floor, None); pi /= pi.sum()
    A = P * (pi / pt); A /= A.sum(1, keepdims=True)
    return A, pi
