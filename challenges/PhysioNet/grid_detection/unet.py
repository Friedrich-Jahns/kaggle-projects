import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet34",        # vortrainiert auf ImageNet
    encoder_weights="imagenet",     # <- HIER: Pretrained
    in_channels=1,                  # z.B. Graustufen
    classes=1                       # 1 Output-Maske
)

x = torch.randn(1, 1, 256, 256)
y = model(x)
print(y.shape)
