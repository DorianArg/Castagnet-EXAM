"""Fabriques des modèles futurs, sans chargement ni entraînement automatique."""

from __future__ import annotations

from torch import nn
from torchvision import models


SUPPORTED_MODELS = (
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
)


def classification_head(model: nn.Module, name: str) -> nn.Module:
    """Retourne uniquement la tête remplacée par le classifieur à quatre classes."""
    if name in {"resnet18", "shufflenet_v2_x1_0"}:
        return model.fc
    if name in {"mobilenet_v3_small", "efficientnet_b0"}:
        return model.classifier
    raise ValueError(f"Architecture inconnue : {name}. Choix : {SUPPORTED_MODELS}")


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Gèle tout sauf la tête, ou dégèle l'intégralité du réseau."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        for parameter in classification_head(model, name).parameters():
            parameter.requires_grad = True


def set_training_phase_mode(model: nn.Module, name: str, head_only: bool) -> None:
    """Évite aussi la mise à jour des statistiques BatchNorm du backbone gelé."""
    if head_only:
        model.eval()
        classification_head(model, name).train()
    else:
        model.train()


def build_pretrained(name: str, num_classes: int = 4, use_weights: bool = True) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_weights else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_weights else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if use_weights else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if use_weights else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Architecture inconnue : {name}. Choix : {SUPPORTED_MODELS}")
    return model
