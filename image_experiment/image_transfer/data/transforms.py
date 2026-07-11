from __future__ import annotations

try:
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    transforms = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_torchvision() -> None:
    if transforms is None:
        raise RuntimeError(f"torchvision is required for image datasets: {_IMPORT_ERROR}")


def build_train_transform(image_size: int):
    require_torchvision()
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])


def build_eval_transform(image_size: int):
    require_torchvision()
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
