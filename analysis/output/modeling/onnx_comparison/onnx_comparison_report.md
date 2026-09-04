# Comparaison ONNX CastagNet

Les metriques de qualite proviennent des resultats validation figes du paragraphe 4.2; elles n'ont pas ete recalculees. Aucun manifeste test n'a ete charge et aucun entrainement n'a ete execute.

## Qualite, taille et cout mesure

| Modele | Accuracy | Macro F1 | Precision Conforme | Recall Conforme | Parametres | ONNX (Mio) | CPU inf. (ms/img/s) | CUDA inf. (ms/img/s) | CPU E2E (ms/img/s) | CUDA E2E (ms/img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 88.15% | 85.21% | 88.36% | 87.82% | 11,178,564 | 42.63 | 17.756 / 56.32 | 3.865 / 258.71 | 36.176 / 27.64 | 19.057 / 52.47 |
| efficientnet_b0 | 86.61% | 83.60% | 87.78% | 83.57% | 4,012,672 | 15.30 | 14.956 / 66.86 | 11.408 / 87.66 | 32.278 / 30.98 | 24.416 / 40.96 |
| mobilenet_v3_small | 84.21% | 80.49% | 87.31% | 80.43% | 1,521,956 | 5.81 | 4.834 / 206.87 | 8.061 / 124.06 | 21.533 / 46.44 | 19.315 / 51.77 |

## Capacite theorique de 12 flux sur la machine mesuree

La comparaison ci-dessous est une division sequentielle simplifiee. Elle ne simule ni concurrence reelle, ni files d'attente, ni capture de 12 flux.

| Modele | Provider | Mesure | 12 img/s | 24 img/s | 60 img/s | 120 img/s | 300 img/s |
|---|---|---|---:|---:|---:|---:|---:|
| resnet18 | CPU | inference_only (56.32 img/s) | OK | OK | NON | NON | NON |
| resnet18 | CPU | end_to_end (27.64 img/s) | OK | OK | NON | NON | NON |
| resnet18 | CUDA | inference_only (258.71 img/s) | OK | OK | OK | OK | NON |
| resnet18 | CUDA | end_to_end (52.47 img/s) | OK | OK | NON | NON | NON |
| efficientnet_b0 | CPU | inference_only (66.86 img/s) | OK | OK | OK | NON | NON |
| efficientnet_b0 | CPU | end_to_end (30.98 img/s) | OK | OK | NON | NON | NON |
| efficientnet_b0 | CUDA | inference_only (87.66 img/s) | OK | OK | OK | NON | NON |
| efficientnet_b0 | CUDA | end_to_end (40.96 img/s) | OK | OK | NON | NON | NON |
| mobilenet_v3_small | CPU | inference_only (206.87 img/s) | OK | OK | OK | OK | NON |
| mobilenet_v3_small | CPU | end_to_end (46.44 img/s) | OK | OK | NON | NON | NON |
| mobilenet_v3_small | CUDA | inference_only (124.06 img/s) | OK | OK | OK | OK | NON |
| mobilenet_v3_small | CUDA | end_to_end (51.77 img/s) | OK | OK | NON | NON | NON |

## Validation numerique PyTorch / ONNX Runtime

L'echantillon comprend 16 indices repartis deterministiquement dans le manifeste validation. La colonne decision 0,55 est un controle de preservation numerique; elle ne calibre pas de seuil pour EfficientNet ou MobileNet.

| Modele | Max logits | Moyenne logits | Max probabilites | Argmax differents | Decisions differentes |
|---|---:|---:|---:|---:|---:|
| resnet18 | 0.000016212 | 0.000004309 | 0.000002086 | 0 | 0 |
| efficientnet_b0 | 0.000931740 | 0.000149926 | 0.000073820 | 0 | 0 |
| mobilenet_v3_small | 0.000210762 | 0.000039052 | 0.000024498 | 0 | 0 |

## Cout du preprocessing

Les mesures preprocessing, inference seule et end-to-end sont des boucles distinctes. Leur moyenne n'est donc pas additive terme a terme. La boucle combinee inclut notamment les allocations, le decodage, les conversions, les copies memoire, l'orchestration Python, les effets de cache et, avec CUDA, les synchronisations imposees par ONNX Runtime. Ces contributions n'ont pas ete instrumentees individuellement; aucune cause unique ne peut etre attribuee. La latence end-to-end reste la reference applicative.

## Conclusion

Sur la RTX 3080 Laptop mesuree, ResNet18 est favorise: il offre la meilleure qualite validation et aussi le meilleur debit CUDA mesure, y compris end-to-end. Sur CPU, MobileNetV3 Small est nettement le plus rapide et le plus compact, au prix d'une baisse de qualite. EfficientNet-B0 constitue un compromis intermediaire sur CPU, mais n'est pas favorise par les mesures CUDA actuelles.

Ces resultats ne valident pas la cible GTX 1060 3 Go / ancien i7. La VRAM par processus n'a pas ete obtenue et aucun vrai essai concurrent a 12 flux n'a ete realise. Aucun chiffre n'est extrapole vers cette cible.
