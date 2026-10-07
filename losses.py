import torch
import torch.nn.functional as F


class diceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, lab):
        numer = (pred*lab).sum(axis=-1).sum(axis=-1)
        denom= pred.sum(axis=-1).sum(axis=-1) + lab.sum(axis=-1).sum(axis=-1) + self.eps
        dloss = 1-2*numer/denom
        dloss = dloss.mean()
        return dloss


class TVLoss(torch.nn.Module):
        def __init__(self, falseP_weight=0.5, eps=1e-8):
            """
            :param falseP_weight: number from 0-1 , weight higher to penalize false positives more
            :param eps: keeps denominator finite
            """
            super().__init__()
            self.eps = eps
            self.alpha = falseP_weight
            self.beta = 1 - falseP_weight

        def forward(self, pred, lab):
            trueP = (pred * lab).sum(axis=-1).sum(axis=-1)
            falseP = (pred * (1-lab)).sum(axis=-1).sum(axis=-1)
            falseN = ((1 - pred) * lab).sum(axis=-1).sum(axis=-1)
            tv_idx = trueP / (trueP + self.alpha*falseP + self.beta*falseN + self.eps)
            loss = 1-tv_idx
            loss = loss.mean()
            return loss


def compute_autocorrelation(x, downsample=4):
    """Wiener-Khinchin autocorrelation via FFT.

    Computes |FFT(x)|^2 -> IFFT -> fftshift, max-normalized.
    Lattice patterns produce sharp periodic peaks in autocorrelation space.
    Adapted from crystal_split/train.py.

    :param x: (B, C, H, W) tensor
    :param downsample: spatial downsample factor before FFT (saves memory/time)
    """
    if downsample > 1:
        x = F.avg_pool2d(x, kernel_size=downsample, stride=downsample)
    B, C, H, W = x.shape
    x_padded = F.pad(x, (W // 2, W // 2, H // 2, H // 2), mode='constant', value=0)
    fft_x = torch.fft.fft2(x_padded)
    autocorr = torch.fft.fftshift(
        torch.fft.ifft2(torch.abs(fft_x) ** 2).real, dim=(-2, -1))
    return autocorr / (autocorr.amax(dim=(-2, -1), keepdim=True) + 1e-8)


class AutocorrStitchLoss(torch.nn.Module):
    """Autocorrelation loss on stitched full-detector images.

    Stitches per-panel predictions into full-detector images, then compares
    their autocorrelation (power spectrum) against ground truth. This loss
    encourages predictions that form lattice-like periodic patterns across
    the full detector, helping suppress false positives (non-periodic noise)
    and recover false negatives (weak peaks that follow the lattice).

    Only used during training; inference remains per-panel.
    """

    def __init__(self, panel_offsets, canvas_h, canvas_w,
                 panels_per_shot=32, stitch_downsample=4):
        """
        :param panel_offsets: (N_panels, 2) array of (ystart, xstart) per panel
        :param canvas_h: full detector height in pixels
        :param canvas_w: full detector width in pixels
        :param panels_per_shot: number of panels per shot (32 for Eiger 16M)
        :param stitch_downsample: downsample stitched image before FFT
        """
        super().__init__()
        self.register_buffer('panel_offsets',
                             torch.tensor(panel_offsets, dtype=torch.long))
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.pps = panels_per_shot
        self.stitch_ds = stitch_downsample

    def _stitch(self, panels, panel_indices):
        """Stitch panels into full-detector image(s).

        :param panels: (N, H, W) panel predictions/labels
        :param panel_indices: (N,) panel position index (0-31)
        :returns: (N_shots, 1, canvas_h, canvas_w) stitched images
        """
        n_shots = panels.shape[0] // self.pps
        ph, pw = panels.shape[1], panels.shape[2]
        canvas = torch.zeros(n_shots, 1, self.canvas_h, self.canvas_w,
                             device=panels.device, dtype=panels.dtype)
        for i in range(panels.shape[0]):
            si = i // self.pps
            pi = panel_indices[i].item()
            y0 = self.panel_offsets[pi, 0].item()
            x0 = self.panel_offsets[pi, 1].item()
            canvas[si, 0, y0:y0 + ph, x0:x0 + pw] = panels[i]
        return canvas

    def forward(self, pred, lab, panel_indices):
        """
        :param pred: (B, H, W) predicted masks (sigmoid output, squeezed)
        :param lab: (B, H, W) ground truth masks
        :param panel_indices: (B,) panel position index for each panel
        :returns: scalar autocorrelation MSE loss
        """
        stitched_pred = self._stitch(pred, panel_indices)
        stitched_lab = self._stitch(lab, panel_indices)
        ac_pred = compute_autocorrelation(stitched_pred, downsample=self.stitch_ds)
        ac_lab = compute_autocorrelation(stitched_lab, downsample=self.stitch_ds)
        return F.mse_loss(ac_pred, ac_lab)