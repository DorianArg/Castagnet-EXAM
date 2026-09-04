# CastagNet

CastagNet est un projet de classification d’images de châtaignes issues d’une
machine de tri. Le pipeline distingue quatre classes : `Conforme`,
`NON Conforme`, `PIETRA` et `Vide`.

Le dépôt rassemble la préparation et l’audit des données (§4.1), la
modélisation et l’évaluation figée (§4.2), puis l’export et les benchmarks ONNX
(§4.3).

## Structure du dépôt

- `analysis/` : qualité des données, revues humaines, appariement Top/Bottom,
  constitution du manifeste final, splits et sorties scientifiques ;
- `training/` : datasets PyTorch, modèles, configurations, entraînement,
  calibration, évaluation finale, exports et benchmarks ONNX ;
- `labeling_tool/` : outil Streamlit de relecture des annotations ;
- `reports/` : rapports finaux des §4.1, §4.2 et §4.3 ;
- `images.dvc` : pointeur DVC vers les 35 254 images historiques ;
- `labels_principal.csv` et `labels_masked.csv` : sources d’annotation
  historiques, non modifiées par les scripts d’analyse.

## Installation

Créer un environnement Python, puis installer les dépendances :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r training/requirements-training.txt
```

Pour récupérer les images historiques, installer DVC avec le support Google
Drive puis utiliser la configuration locale de l’équipe :

```powershell
python -m pip install "dvc[gdrive]"
dvc pull
```

Le fichier `.dvc/config` peut contenir des identifiants locaux et n’est jamais
versionné. `.dvc/config.example` documente uniquement la structure attendue.

## Données

Les images sont attendues par défaut dans `<repo_root>/images`. Elles sont
gérées avec DVC et ignorées par Git. Les commandes d’entraînement et
d’évaluation acceptent `--image-root` pour utiliser un autre emplacement.

Les frames et crops manuels du §4.1 se trouvent sous
`analysis/extracted_video/`. Ils sont volontairement versionnables dans Git et
ne doivent pas faire l’objet d’un `dvc add` ou d’un `dvc push` vers le Drive
partagé.

La cible finale est exclusivement `effective_label`. `label_filename` reste
une information historique et n’est jamais utilisé pour déterminer la classe
d’entraînement.

## §4.1 — Qualité et préparation des données

Les scripts d’`analysis/` construisent un manifeste dérivé, documentent les
annotations, exportent les cas ambigus, réalisent les revues humaines et
auditent l’appariement Top/Bottom. Aucun CSV source n’est modifié.

```powershell
python analysis/dataset_quality.py
python analysis/tb_pairing_tool.py --audit
python analysis/build_final_training_manifest.py
```

Le manifeste final est `analysis/output/final_training_manifest.csv`.

## Splits figés

Les splits group-aware utilisés pendant toute l’étude sont :

- `analysis/output/train_manifest.csv` ;
- `analysis/output/validation_manifest.csv` ;
- `analysis/output/test_manifest.csv`.

Le test n’a été ouvert qu’une fois, après sélection du modèle et fixation du
seuil sur validation.

## §4.2 — Entraînement et évaluation

Le protocole complet est décrit dans `training/README.md`. Les configurations
sont dans `training/configs/` et les résultats portables dans
`analysis/output/modeling/`.

```powershell
python training/train.py --smoke-test --config training/configs/resnet18.json
```

Le modèle final est un ResNet18 pré-entraîné ImageNet, sélectionné sur
validation. Le checkpoint retenu correspond à l’epoch 13 du run MLflow
`db0ee7873903411e899e2f85384ae58a`.

La règle métier figée est : prédire `Conforme` si `P(Conforme) >= 0.55`, sinon
choisir l’argmax parmi `NON Conforme`, `PIETRA` et `Vide`. Le seuil a été choisi
sur validation avant l’évaluation finale.

## MLflow

`mlflow.db` contient les runs et métriques de l’étude. Certains `artifact_uri`
historiques correspondent à la machine de développement ; les artefacts finaux
portables sont disponibles dans `analysis/output/`.

Les dossiers locaux `mlartifacts/` et `mlruns/` ne sont pas versionnés.

## §4.3 — ONNX et benchmarks

Les exports ONNX finaux de ResNet18, EfficientNet-B0 et MobileNetV3 Small sont
conservés sous `analysis/output/modeling/<modele>/onnx/`. ResNet18 possède une
version batch 1 statique et une version à batch dynamique.

```powershell
python training/export_onnx.py --model resnet18
python training/benchmark_onnx.py --model resnet18 --provider all
python training/benchmark_dynamic_batch.py
```

Les benchmarks couvrent CPU, CUDA, le temps d’inférence, la chaîne end-to-end
et le comportement par batch. L’étude de batch dynamique documente notamment
le dimensionnement de 12 flux concurrents.

## Résultats principaux

ResNet18 sur le test final avec le seuil figé à 0.55 :

- précision `Conforme` : **89,16 %** ;
- rappel `Conforme` : **85,54 %** ;
- macro-F1 : **84,44 %**.

ONNX sur la machine de développement équipée d’une **RTX 3080 Laptop** :

- inférence CUDA batch 1 : environ **3,87 ms** ;
- end-to-end CUDA batch 1 : environ **19,06 ms** ;
- meilleur débit end-to-end : environ **58,04 images/s** au batch 8.

La GTX 1060 3 Go n’a pas été benchmarkée directement. Aucun de ces résultats
ne doit être interprété comme une mesure effectuée sur GTX 1060.

## Rapports finaux

- `reports/Rapport_CastagNet_4_1_Qualite_Dataset.docx` et `.pdf` ;
- `reports/Rapport_CastagNet_4_2_Modelisation.docx` et `.pdf` ;
- `reports/Rapport_CastagNet_4_3_ONNX_Embarquement_FINAL.docx` et `.pdf`.

Les CSV, JSON, figures et matrices nécessaires à la traçabilité des chiffres
restent disponibles dans `analysis/output/`.
