"""
Training losses for semiconductor image restoration.

Important:
- All numerically sensitive loss calculations run in FP32.
- Input degraded images are not clipped before L1/FFT.
- SSIM-family terms use [0,1] because SSIM is defined for normalized images.
- MS-SSIM implementation uses contrast-structure terms at intermediate
  scales and full SSIM at the final scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coords = (
        torch.arange(kernel_size, dtype=torch.float32)
        - kernel_size // 2
    )
    g = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    padding: int,
):
    """
    Returns:
        ssim_mean
        cs_mean

    Computed in FP32 for stability.
    """
    x = x.float()
    y = y.float()
    k = kernel.float().to(x.device)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.conv2d(x, k, padding=padding)
    mu_y = F.conv2d(y, k, padding=padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x = F.conv2d(x * x, k, padding=padding) - mu_x2
    sig_y = F.conv2d(y * y, k, padding=padding) - mu_y2
    sig_xy = F.conv2d(x * y, k, padding=padding) - mu_xy

    # Floating-point noise can produce tiny negative variances.
    sig_x = sig_x.clamp_min(0.0)
    sig_y = sig_y.clamp_min(0.0)

    numerator = (
        (2.0 * mu_xy + C1)
        * (2.0 * sig_xy + C2)
    )

    denominator = (
        (mu_x2 + mu_y2 + C1)
        * (sig_x + sig_y + C2)
    )

    ssim_map = numerator / denominator.clamp_min(1e-12)

    # SSIM is conventionally interpreted in [0,1] here.
    ssim_map = ssim_map.clamp(0.0, 1.0)

    # Contrast-structure term.
    cs_map = (
        2.0 * sig_xy + C2
    ) / (
        sig_x + sig_y + C2
    )
    cs_map = cs_map.clamp(0.0, 1.0)

    return ssim_map.mean(), cs_map.mean()


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(
            pred.float(),
            target.float(),
        )


class MSSSIMLoss(nn.Module):
    """
    True 3-scale MS-SSIM.

    For intermediate scales:
        contrast-structure term

    Final scale:
        full SSIM term
    """

    def __init__(
        self,
        n_scales: int = 3,
        kernel_size: int = 11,
        sigma: float = 1.5,
    ):
        super().__init__()

        if n_scales < 2:
            raise ValueError(
                "MS-SSIM requires at least two scales."
            )

        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        kernel = _gaussian_kernel(
            kernel_size,
            sigma,
        )

        self.register_buffer(
            "kernel",
            kernel,
        )

        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            dtype=torch.float32,
        )[:n_scales]

        # Renormalize because only a subset of the canonical 5 levels
        # is used for 128x128 images.
        weights = weights / weights.sum()

        self.register_buffer(
            "weights",
            weights,
        )

    def forward(self, pred, target):

        x = pred.float().clamp(0.0, 1.0)
        y = target.float().clamp(0.0, 1.0)

        values = []

        for scale in range(self.n_scales):

            ssim_value, cs_value = _ssim_components(
                x,
                y,
                self.kernel,
                self.padding,
            )

            if scale == self.n_scales - 1:
                values.append(ssim_value)
            else:
                values.append(cs_value)

            if scale != self.n_scales - 1:
                x = F.avg_pool2d(
                    x,
                    kernel_size=2,
                    stride=2,
                )
                y = F.avg_pool2d(
                    y,
                    kernel_size=2,
                    stride=2,
                )

        values = torch.stack(values)
        values = values.clamp_min(1e-8)

        ms_ssim = torch.exp(
            torch.sum(
                self.weights.to(values.device)
                * torch.log(values)
            )
        )

        return 1.0 - ms_ssim


class SSIMLoss(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
    ):
        super().__init__()

        self.register_buffer(
            "kernel",
            _gaussian_kernel(
                kernel_size,
                sigma,
            ),
        )

        self.padding = kernel_size // 2

    def forward(self, pred, target):
        x = pred.float().clamp(0.0, 1.0)
        y = target.float().clamp(0.0, 1.0)

        ssim, _ = _ssim_components(
            x,
            y,
            self.kernel,
            self.padding,
        )

        return 1.0 - ssim


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=torch.float32,
        )

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=torch.float32,
        )

        self.register_buffer(
            "sobel_x",
            sobel_x[None, None],
        )
        self.register_buffer(
            "sobel_y",
            sobel_y[None, None],
        )

    def _edges(self, x):
        x = x.float().clamp(0.0, 1.0)

        gx = F.conv2d(
            x,
            self.sobel_x.float(),
            padding=1,
        )

        gy = F.conv2d(
            x,
            self.sobel_y.float(),
            padding=1,
        )

        return torch.sqrt(
            gx * gx + gy * gy + 1e-8
        )

    def forward(self, pred, target):
        return F.l1_loss(
            self._edges(pred),
            self._edges(target),
        )


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        p = torch.fft.fft2(
            pred,
            norm="ortho",
        )

        t = torch.fft.fft2(
            target,
            norm="ortho",
        )

        return F.l1_loss(
            torch.abs(p),
            torch.abs(t),
        )


class VGGPerceptualLoss(nn.Module):
    """
    VGG19 relu2_2 + relu3_3.

    This is a perceptual feature loss, not the LPIPS algorithm itself.
    It is therefore treated as an optional training component and should
    be validated through ablation.
    """

    def __init__(self):
        super().__init__()

        try:
            import torchvision.models as tv
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for VGG perceptual loss."
            ) from exc

        try:
            vgg = tv.vgg19(
                weights=tv.VGG19_Weights.IMAGENET1K_V1
            ).features
        except Exception as exc:
            import logging
            logging.getLogger("semicon.losses").warning(
                "Could not fetch online VGG19 weights (%s); using local features.", exc
            )
            vgg = tv.vgg19(weights=None).features

        # relu2_2 is index 8.
        self.slice1 = vgg[:9]

        # pool2 + conv3_1 + relu3_1 + conv3_2 + relu3_2
        # + conv3_3 + relu3_3 -> index 15.
        self.slice2 = vgg[9:16]

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

    def _preprocess(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise AssertionError(
                "VGGPerceptualLoss expects single-channel "
                f"input [B,1,H,W], got {tuple(x.shape)}"
            )

        x = x.float().clamp(0.0, 1.0)
        x = x.repeat(1, 3, 1, 1)

        return (
            x - self.mean.float()
        ) / self.std.float()

    def forward(self, pred, target):

        p = self._preprocess(pred)
        t = self._preprocess(target)

        p1 = self.slice1(p)
        t1 = self.slice1(t)

        p2 = self.slice2(p1)
        t2 = self.slice2(t1)

        return (
            F.l1_loss(p1, t1)
            + F.l1_loss(p2, t2)
        )


class RestorationLoss(nn.Module):
    """
    Combined restoration loss.

    Optional l1_weights must be [B], normalized to mean 1 by the caller.
    This enables true hard-example weighting without changing the effective
    global loss scale.
    """

    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.10,
        w_perceptual=0.10,
        w_fft=0.05,
        w_edge=0.05,
        use_msssim=True,
    ):
        super().__init__()

        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)
        self.w_perc = float(w_perceptual)
        self.w_fft = float(w_fft)
        self.w_edge = float(w_edge)

        self.l1 = L1Loss()

        self.ssim = None
        if self.w_ssim > 0:
            self.ssim = (
                MSSSIMLoss()
                if use_msssim
                else SSIMLoss()
            )

        self.vgg = None
        if self.w_perc > 0:
            self.vgg = VGGPerceptualLoss()

        self.fft = (
            FFTLoss()
            if self.w_fft > 0
            else None
        )

        self.edge = (
            SobelEdgeLoss()
            if self.w_edge > 0
            else None
        )

    @staticmethod
    def per_sample_l1(pred, target):
        return torch.mean(
            torch.abs(
                pred.float() - target.float()
            ),
            dim=(1, 2, 3),
        )

    def forward(
        self,
        pred,
        target,
        l1_weights=None,
        precomputed_l1=None,
    ):
        pred = pred.float()
        target = target.float()

        losses = {}
        total = pred.new_tensor(0.0)

        if self.w_l1 > 0:
            per_sample = self.per_sample_l1(
                pred,
                target,
            )

            if l1_weights is None:
                l1 = per_sample.mean()
            else:
                if l1_weights.shape != per_sample.shape:
                    raise ValueError(
                        "l1_weights must have shape [B]."
                    )
                l1 = (
                    per_sample
                    * l1_weights.float()
                ).mean()

            losses["l1"] = l1
            total = total + self.w_l1 * l1

        if self.ssim is not None:
            losses["ssim"] = self.ssim(
                pred,
                target,
            )
            total = total + (
                self.w_ssim
                * losses["ssim"]
            )

        if self.fft is not None:
            losses["fft"] = self.fft(
                pred,
                target,
            )
            total = total + (
                self.w_fft
                * losses["fft"]
            )

        if self.edge is not None:
            losses["edge"] = self.edge(
                pred,
                target,
            )
            total = total + (
                self.w_edge
                * losses["edge"]
            )

        if self.vgg is not None:
            losses["perceptual"] = self.vgg(
                pred,
                target,
            )
            total = total + (
                self.w_perc
                * losses["perceptual"]
            )

        losses["total"] = total
        return total, losses
