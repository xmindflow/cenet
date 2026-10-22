"""PyTorch Hub entry points for CENet.

    import torch
    model = torch.hub.load("xmindflow/cenet", "cenet", dataset="synapse")
    predictor = torch.hub.load("xmindflow/cenet", "predictor", dataset="ham10000")
    mask = predictor.predict("lesion.jpg")

Weights are downloaded from https://huggingface.co/afshinbigboy/CENet.
"""
import os
import sys

dependencies = ["torch", "timm", "monai", "numpy", "scipy", "huggingface_hub"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import torch.nn.functional as F

HF_REPO = "afshinbigboy/CENet"
IMG_SIZE = 224

_COMMON = dict(encoder="pvt_v2_b2", skip_mode="cat", dec_up_block="eucb",
               out_merge_mode="cat", out_up_block="upcn", out_up_ks=3)

CONFIGS = {
    "synapse": dict(file="synapse_2418962.pth", input_channels=1, num_classes=9,
                    scale_factors=[1.0, 0.75, 0.5], diffatt_num_heads=[16, 8, 8],
                    classes=["background", "aorta", "gallbladder", "left kidney", "right kidney",
                             "liver", "pancreas", "spleen", "stomach"]),
    "acdc": dict(file="acdc_2418732.pth", input_channels=1, num_classes=4,
                 scale_factors=[1.0, 0.5], diffatt_num_heads=[4, 4, 4],
                 classes=["background", "right ventricle", "myocardium", "left ventricle"]),
    "ham10000": dict(file="ham10k_2418602.pth", input_channels=3, num_classes=2,
                     scale_factors=[1.0, 0.75, 0.5], diffatt_num_heads=[2, 2, 2],
                     classes=["background", "lesion"]),
    "ph2": dict(file="ph2_2418571.pth", input_channels=3, num_classes=2,
                scale_factors=[1.0, 0.75, 0.5], diffatt_num_heads=[2, 2, 2],
                classes=["background", "lesion"]),
}


def _load_weights(net, filename, map_location):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO, filename=filename)
    net.load_state_dict(torch.load(path, map_location=map_location or "cpu", weights_only=True))
    return net


def cenet(dataset="synapse", pretrained=True, map_location=None):
    """CENet for `dataset` in {"synapse", "acdc", "ham10000", "ph2"}. Returns the model in eval mode.

    Input: (B, C, 224, 224) float tensor, C=1 for synapse/acdc and C=3 for ham10000/ph2.
    Output: (B, num_classes, 224, 224) logits.
    """
    from networks import CENet
    if dataset not in CONFIGS:
        raise ValueError(f"dataset must be one of {list(CONFIGS)}, got {dataset!r}")
    cfg = CONFIGS[dataset]
    net = CENet(input_channels=cfg["input_channels"], num_classes=cfg["num_classes"],
                scale_factors=cfg["scale_factors"], diffatt_num_heads=cfg["diffatt_num_heads"],
                enc_pretrain=False, **_COMMON)
    if pretrained:
        _load_weights(net, cfg["file"], map_location)
    return net.eval()


def cenet_org(pretrained=True, map_location=None):
    """The original CENet model reported in the paper, trained on Synapse (9 classes, 1-channel input)."""
    from networks import CENetOrg
    net = CENetOrg(num_classes=9, input_channels=1, scale_factors=[0.8, 0.4],
                   num_heads=[16, 8, 8], pretrain=False)
    if pretrained:
        _load_weights(net, "synapse_cenet_org_2352272.pth", map_location)
    return net.eval()


class CENetPredictor:
    """Wraps a CENet model with the preprocessing used during training.

    `predict` takes a file path, PIL image or numpy array and returns an integer mask
    with the same height and width as the input.
      - ham10000 / ph2: an RGB image (H, W, 3).
      - synapse / acdc: a 2D slice (H, W) or a volume (D, H, W), predicted slice by slice.
    """

    def __init__(self, dataset="synapse", device=None):
        self.dataset = dataset
        self.cfg = CONFIGS[dataset]
        self.classes = self.cfg["classes"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = cenet(dataset).to(self.device)

    @staticmethod
    def _read(image):
        if isinstance(image, (str, os.PathLike)):
            path = str(image)
            if path.endswith((".nii", ".nii.gz")):
                import SimpleITK as sitk
                return sitk.GetArrayFromImage(sitk.ReadImage(path))  # (D, H, W)
            if path.endswith(".npy"):
                return np.load(path)
            from PIL import Image
            image = Image.open(path)
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        return image

    @torch.no_grad()
    def _run(self, x):
        """x: (N, C, 224, 224) float tensor -> (N, 224, 224) labels."""
        out = self.model(x.to(self.device))
        return out.argmax(dim=1).cpu()

    def _predict_skin(self, img):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3]
        h, w = img.shape[:2]
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None]
        if img.dtype == np.uint8 or x.max() > 1:
            x = x / 255.0
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        pred = self._run(x)[None].float()
        return F.interpolate(pred, size=(h, w), mode="nearest")[0, 0].numpy().astype(np.uint8)

    def _scale_gray(self, vol, raw_hu):
        vol = vol.astype(np.float32)
        if self.dataset == "synapse":
            if raw_hu is None:
                raw_hu = vol.min() < 0
            if raw_hu:  # CT in Hounsfield units: same window as the Synapse training data
                return (np.clip(vol, -125, 275) + 125) / 400
        if vol.max() > 1 or vol.min() < 0:
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol

    def _predict_gray(self, vol, raw_hu):
        from scipy.ndimage import zoom
        if vol.ndim == 3 and vol.shape[-1] in (3, 4):  # RGB(A) image of a slice
            vol = vol[..., :3].mean(-1)
        is_2d = vol.ndim == 2
        vol = self._scale_gray(vol[None] if is_2d else vol, raw_hu)
        d, h, w = vol.shape
        slices = np.stack([zoom(s, (IMG_SIZE / h, IMG_SIZE / w), order=3) for s in vol])
        x = torch.from_numpy(slices).float()[:, None]
        if self.dataset == "synapse":
            x = (x - 0.5) / 0.5
        preds = torch.cat([self._run(x[i:i + 16]) for i in range(0, d, 16)]).numpy()
        out = np.stack([zoom(p, (h / IMG_SIZE, w / IMG_SIZE), order=0) for p in preds]).astype(np.uint8)
        return out[0] if is_2d else out

    def predict(self, image, raw_hu=None):
        """Return the label mask. For synapse, `raw_hu` says whether the input is CT in
        Hounsfield units (auto-detected from negative values when None)."""
        img = self._read(image)
        if self.dataset in ("ham10000", "ph2"):
            return self._predict_skin(img)
        return self._predict_gray(img, raw_hu)


def predictor(dataset="synapse", device=None):
    """A CENetPredictor: `predictor.predict(image)` handles preprocessing and returns a mask."""
    return CENetPredictor(dataset, device)
