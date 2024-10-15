
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.ops import knn_points


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class GaussianRegularization:
    
    regularizations: dict

    @torch.no_grad()
    def prepare_knn_vectors_and_norms(self, vertices, K=1, eps=1e-12):
        """
        Args:
            vertices: torch.Tensor, [V, 3]
        """
        knn_outputs = knn_points(vertices[None], vertices[None], K=K+1, norm=2)
        # knn_dists = knn_outputs.dists[0, :, 1:]  # [V, K]
        knn_indices = knn_outputs.idx[0, :, 1:]  # [V, K]
        knn_vertices = vertices[knn_indices, :]  # [V, K, 3]

        knn_vectors = knn_vertices - vertices.unsqueeze(1)  # [V, K, 3]
        knn_vectors_of_points = knn_vectors[self.vertex_indices]  # [N, K, 3]

        norms_of_knn_vectors = torch.norm(knn_vectors, p=2, dim=-1, keepdim=False)  # [V, K]
        norms_of_knn_vectors_of_points = norms_of_knn_vectors.clamp_min(eps)[self.vertex_indices]  # [N, K]

        # normalized_knn_vectors = torch.nn.functional.normalize(knn_vectors, dim=-1)  # [V, K, 3]
        # normalized_knn_vectors_of_points = normalized_knn_vectors[self.vertex_indices]  # [N, K, 3]

        self.regularizations['K'] = K
        self.regularizations['knn_vectors_of_points'] = knn_vectors_of_points
        self.regularizations['norms_of_knn_vectors_of_points'] = norms_of_knn_vectors_of_points

    def compute_offset_regularization_loss(self, offsets):
        """
        Args:
            offsets: torch.Tensor, [N, 3]
        """
        K = self.regularizations['K']
        knn_vectors_of_points = self.regularizations['knn_vectors_of_points']  # [N, K, 3]
        norms_of_knn_vectors_of_points = self.regularizations['norms_of_knn_vectors_of_points']  # [N, K]
        
        point_vectors = offsets.unsqueeze(1).expand(-1, K, -1)  # [N, K, 3]
        projections = torch.linalg.vecdot(point_vectors, knn_vectors_of_points) / norms_of_knn_vectors_of_points  # [N, K]

        errors = (projections / norms_of_knn_vectors_of_points - 0.5).clamp_min(0.0)  # [N, K]
        return errors.mean(dim=-1).sum()

    def compute_scale_regularization_loss(self, scales):
        """
        Args:
            scales: torch.Tensor, [N, 3]
        """
        K = self.regularizations['K']
        norms_of_knn_vectors_of_points = self.regularizations['norms_of_knn_vectors_of_points']  # [N, K]

        scales = scales.unsqueeze(1).expand(-1, K, -1).max(dim=-1).values  # [N, K]
        errors = (scales / norms_of_knn_vectors_of_points - 1.0).clamp_min(0.0)  # [N, K]
        return errors.mean(dim=-1).sum()

    def forward(self, body_outputs):
        gaussians = super().forward()
        if self.training:
            self.prepare_knn_vectors_and_norms(vertices=body_outputs['mesh'].v)

            offset_regularization_loss = self.compute_offset_regularization_loss(self._offsets)
            self.regularizations['offset_regularization_loss'] = offset_regularization_loss

            scale_regularization_loss = self.compute_scale_regularization_loss(gaussians[-1])
            self.regularizations['scale_regularization_loss'] = scale_regularization_loss
        return gaussians


class ImageReconstructionLoss:
    def __init__(self, lambda_dssim:float=0.2):
        self.lambda_dssim = lambda_dssim

    def __call__(self, image:torch.Tensor, gt_image:torch.Tensor):
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim(image, gt_image))
        return loss
