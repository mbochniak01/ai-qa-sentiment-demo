from typing import Callable, List

import pandas as pd


def check_class_balance(
        preds: pd.Series,
        min_ratio: float = 0.3,
        max_ratio: float = 0.7
) -> None:
    """Ensure predicted classes are reasonably balanced."""
    counts = preds.value_counts(normalize=True)
    for cls, freq in counts.items():
        assert min_ratio <= freq <= max_ratio, f"Class '{cls}' is imbalanced ({freq:.2f})"


def apply_perturbations(samples: List[str], perturbations: List[Callable[[str], str]]) -> List[str]:
    """Apply perturbation functions to input samples."""
    perturbed = []
    for sample in samples:
        perturbed.extend([pert(sample) for pert in perturbations])
    return perturbed
