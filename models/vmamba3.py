from config.constants import *
from models.vmamba import VMambaClassifier
from models.vmamba3_backbone import VisualMamba3


class VMamba3Classifier(VMambaClassifier):
    """End-to-end classifier identical to VMambaClassifier but with a Mamba-3 backbone."""

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        ssm_dim=SSM_DIM,
        expand_dim=EXPAND_DIM,
        lr=LR,
        class_weights=None,
        pretrained=False,
    ):
        super().__init__(
            num_classes=num_classes, img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, ssm_dim=ssm_dim, expand_dim=expand_dim,
            lr=lr, class_weights=class_weights, pretrained=pretrained,
        )
        self.backbone = VisualMamba3(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depth=depth, ssm_dim=ssm_dim, expand_dim=expand_dim,
            learning_rate=lr, mask_ratio=0.0, use_cls_token=False,
        )
