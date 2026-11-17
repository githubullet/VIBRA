"""The retrieval ranks agree with a brute force count on random features with exact score ties and several positives per query."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vibra.data import positives
from vibra.evaluate import ranks


def brute(Q, G, pos_lists):
    S = (Q @ G.T).numpy()
    return np.array([int((S[i] > max(S[i, p] for p in ps)).sum()) for i, ps in enumerate(pos_lists)])


def test_ranks():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n_img = 60
        cap_image_ids = np.concatenate([[f"im{i}"] * c for i, c in enumerate(rng.integers(1, 6, n_img))])
        rng.shuffle(cap_image_ids)
        image_ids = np.array([f"im{i}" for i in range(n_img)])
        gi = torch.tensor(rng.integers(-2, 3, (n_img, 8)), dtype=torch.float32)
        gt = torch.tensor(rng.integers(-2, 3, (len(cap_image_ids), 8)), dtype=torch.float32)
        cap2img, img2caps = positives(image_ids, cap_image_ids)
        assert (ranks(gi, gt, torch.as_tensor(img2caps)) == brute(gi, gt, [list(r[r >= 0]) for r in img2caps])).all()
        assert (ranks(gt, gi, torch.as_tensor(cap2img)[:, None]) == brute(gt, gi, [[c] for c in cap2img])).all()


if __name__ == "__main__":
    test_ranks()
    print("ok")
