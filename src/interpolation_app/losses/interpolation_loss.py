import torch.nn as nn
import kornia.losses as kornia_losses
import torchvision.models as models
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    def __init__(self, layer: int = 16):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:layer]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        pred = self.transform(pred)
        target = self.transform(target)
        return nn.functional.l1_loss(self.vgg(pred), self.vgg(target))


def get_loss_fn(loss_type: str = "l1"):
    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "ssim":
        return kornia_losses.SSIMLoss(window_size=11, reduction="mean")
    elif loss_type == "perceptual":
        return PerceptualLoss()
    elif loss_type == "mixed":
        l1 = nn.L1Loss()
        ssim = kornia_losses.SSIMLoss(window_size=11, reduction="mean")
        return lambda pred, target: 0.5 * l1(pred, target) + 0.5 * ssim(pred, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
