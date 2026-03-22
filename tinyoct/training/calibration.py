"""
Temperature Scaling — Post-hoc confidence calibration.
Fits a single scalar T on the validation set after training.
Expected Calibration Error (ECE) should decrease after calibration.

Reference: Guo et al., ICML 2017 — 'On Calibration of Modern Neural Networks'
Paper stat to report: ECE before and after calibration.
"""

import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """
    Learns a single temperature scalar to calibrate logits.
    Fit on val set; evaluate ECE on test set.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Re-use the model's log_temperature parameter
        self.model.log_temperature.requires_grad_(True)

    def fit(self, val_loader, device):
        """Fit temperature on validation set using LBFGS."""
        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                all_logits.append(logits)
                all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.model.log_temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            T = self.model.log_temperature.exp()
            scaled = logits / T
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        T_final = self.model.log_temperature.exp().item()
        print(f"Temperature calibrated: T = {T_final:.4f}")
        # Freeze after calibration
        self.model.log_temperature.requires_grad_(False)
        return T_final

    @staticmethod
    def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        """Expected Calibration Error."""
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        for b in range(n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.any():
                acc_bin  = accuracies[mask].float().mean()
                conf_bin = confidences[mask].mean()
                ece += mask.float().mean() * (conf_bin - acc_bin).abs()

        return ece.item()
