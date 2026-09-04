# Pipeline d'entraînement CastagNet

Le dossier `training/` annoncé dans le sujet était absent du dépôt remis. Cette
infrastructure minimale a donc été créée pour exploiter les splits figés du §4.1.
Les quatre architectures pré-entraînées préparées ici (ResNet18, MobileNetV3
Small, EfficientNet-B0 et ShuffleNetV2 x1.0) ont été choisies dans ce travail ;
elles ne proviennent pas d'un matériel officiel retrouvé et ne sont pas lancées
pendant l'étape baseline.

## Règles de données

- cible unique : `effective_label` ;
- mapping explicite : Conforme=0, NON Conforme=1, PIETRA=2, Vide=3 ;
- train : `analysis/output/train_manifest.csv` ;
- validation : `analysis/output/validation_manifest.csv` ;
- le manifeste test est interdit dans `training/train.py` ;
- les images sont résolues par défaut dans `<repo_root>/images`, avec override
  possible via `--image-root`.

## Prétraitement

Les images sont converties en RGB, redimensionnées en conservant leur ratio puis
complétées par un padding noir jusqu'à 224×224. Elles sont converties en tenseur
et normalisées avec les statistiques ImageNet. Train ajoute uniquement un flip
horizontal de probabilité 0,5 et une rotation uniforme entre -10° et +10° ; la
validation est entièrement déterministe.

## Commandes

Depuis la racine du dépôt :

```powershell
python training/train.py --smoke-test
python training/train.py --train
```

La machine GPU validée utilise les wheels officielles PyTorch CUDA 12.6 :

```powershell
python -m pip install --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

Le smoke test vérifie les DataLoaders, forward, logits `[batch, 4]`, loss,
backward, optimizer, absence de NaN, checkpoint local, MLflow et débit CPU/GPU.
Le full training refuse de démarrer si l'estimation enregistrée par le smoke test
dépasse `max_estimated_training_hours` (2 h par défaut), sauf décision explicite
future matérialisée par `--allow-slow-training`.

## Transfer learning

Les quatre configurations torchvision utilisent les poids ImageNet officiels et
un protocole commun dans un seul run MLflow : trois epochs avec backbone gele et
tete seule a `lr=0.001`, puis degel complet pour au plus quinze epochs a
`lr=0.0001`. La seconde phase emploie AdamW, CrossEntropyLoss non ponderee,
ReduceLROnPlateau et un early stopping de patience 5. La colonne `phase` de
`training_history.csv` distingue `head` et `finetune`.

Exemple de smoke test ResNet18, sans entrainement complet :

```powershell
python training/train.py --smoke-test `
  --config training/configs/resnet18.json `
  --output-dir analysis/output/modeling/resnet18/
```

## MLflow

Tracking local SQLite dans `mlflow.db`, artefacts dans `mlartifacts/`, expérience `CastagNet_4_2`. Le smoke test utilise
`custom_cnn_baseline_v1_smoke`; le run complet utilise
`custom_cnn_baseline_v1`. Le jeu test n'est jamais chargé.

## Reproductibilité

Les seeds Python, NumPy et PyTorch sont fixés. Les algorithmes déterministes sont
activés quand possible. Une reproductibilité bit-à-bit n'est toutefois pas
garantie sur tous les matériels CUDA.
