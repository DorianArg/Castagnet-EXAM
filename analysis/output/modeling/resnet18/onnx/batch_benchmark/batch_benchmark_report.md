# Benchmark ONNX ResNet18 a batch dynamique

Export du checkpoint fige epoch 13, opset 17, logits uniquement. Validation et benchmark utilisent exclusivement des images du manifeste validation.

## Validation numerique

| Batch | Max ecart logits | Ecart moyen logits | Max ecart probabilites | Argmax differents | Decisions seuil 0.55 differentes |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000008106 | 0.000005186 | 0.000000018 | 0 | 0 |
| 4 | 0.000008106 | 0.000002645 | 0.000000268 | 0 | 0 |
| 12 | 0.000017166 | 0.000004047 | 0.000002041 | 0 | 0 |

## Benchmark CUDA

| Batch | Inf. moyenne ms/batch | Inf. p95 | Inf. ms/image | Inf. img/s | E2E moyenne ms/batch | E2E p95 | E2E ms/image | E2E img/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.115 | 5.084 | 4.115 | 243.02 | 20.473 | 24.539 | 20.473 | 48.85 |
| 2 | 4.976 | 6.802 | 2.488 | 401.95 | 51.727 | 63.471 | 25.864 | 38.66 |
| 4 | 5.806 | 8.748 | 1.452 | 688.93 | 90.512 | 103.833 | 22.628 | 44.19 |
| 8 | 6.250 | 7.660 | 0.781 | 1280.04 | 137.826 | 165.108 | 17.228 | 58.04 |
| 12 | 7.854 | 8.369 | 0.654 | 1527.94 | 237.794 | 261.488 | 19.816 | 50.46 |

## Etapes isolees

| Batch | Decodage ms/batch | Preprocessing sans decodage ms/batch | Constitution batch ms/batch | session.run ms/batch |
|---:|---:|---:|---:|---:|
| 1 | 6.462 | 5.712 | 0.100 | 4.115 |
| 2 | 16.455 | 15.540 | 0.962 | 4.976 |
| 4 | 35.638 | 34.134 | 1.729 | 5.806 |
| 8 | 41.310 | 48.367 | 2.705 | 6.250 |
| 12 | 61.586 | 80.368 | 5.858 | 7.854 |

## Capacite theorique pour 12 cameras

| Batch | Mesure | 12 img/s | 24 img/s | 60 img/s | 120 img/s | 300 img/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Inference seule (243.02 img/s) | OK | OK | OK | OK | NON |
| 1 | End-to-end (48.85 img/s) | OK | OK | NON | NON | NON |
| 2 | Inference seule (401.95 img/s) | OK | OK | OK | OK | OK |
| 2 | End-to-end (38.66 img/s) | OK | OK | NON | NON | NON |
| 4 | Inference seule (688.93 img/s) | OK | OK | OK | OK | OK |
| 4 | End-to-end (44.19 img/s) | OK | OK | NON | NON | NON |
| 8 | Inference seule (1280.04 img/s) | OK | OK | OK | OK | OK |
| 8 | End-to-end (58.04 img/s) | OK | OK | NON | NON | NON |
| 12 | Inference seule (1527.94 img/s) | OK | OK | OK | OK | OK |
| 12 | End-to-end (50.46 img/s) | OK | OK | NON | NON | NON |

## Portee des mesures

Le decodage, le preprocessing sans decodage et la constitution NumPy du batch sont mesures separement. `session.run` regroupe la copie de l'entree CPU vers le GPU, l'execution CUDA, la synchronisation et le retour de la sortie; la copie CPU-GPU n'est donc pas isolee.

La mesure end-to-end (decodage + preprocessing + batch + session.run) est la reference applicative. Un batch de 12 mesure uniquement le potentiel de mutualisation du GPU et ne simule pas douze flux video concurrents.

## Limites

Mesures realisees sur RTX 3080 Laptop 8 Go. Aucune extrapolation numerique n'est faite vers la GTX 1060 3 Go / ancien i7. Le test ne couvre ni capture video, ni files d'attente, ni ordonnancement de 12 flux, ni contention memoire d'une application complete.
