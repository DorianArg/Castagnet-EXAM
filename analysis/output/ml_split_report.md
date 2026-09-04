# Splits ML CastagNet — §4.2

## Objectif

Les splits sont construits avant tout entraînement afin de figer une base commune à toutes les expériences. `train` sert à apprendre les paramètres ; `validation` sert à comparer les architectures, régler les hyperparamètres, surveiller l’overfitting et sélectionner le modèle ; `test` est réservé à l’évaluation finale. Le test est figé par ce manifeste et ne doit pas servir au choix d’architecture, learning rate, augmentation, nombre d’epochs, seuil ou modèle.

## Choix 70/15/15

Le compromis 70/15/15 réserve la majorité des données à l’apprentissage tout en maintenant plusieurs milliers d’images pour la sélection et l’évaluation finales. Ce ratio n’est pas universel ; il est adapté au volume disponible. Les proportions portent sur les images, avec de petits écarts admis pour préserver les groupes indivisibles.

| split | images | % réel | split_group_id uniques |
| --- | --- | --- | --- |
| train | 24677 | 69.9977 % | 13333 |
| validation | 5289 | 15.0026 % | 2877 |
| test | 5288 | 14.9997 % | 2868 |

## Risque de fuite T/B

Un `random_split` au niveau image pourrait placer les deux vues d’une même châtaigne potentielle dans des jeux différents. L’affectation se fait donc exclusivement au niveau de `split_group_id`. Les associations historiques restent heuristiques, mais une fois admises elles sont conservées intégralement dans un seul split. Les 5 455 paires dont les labels diffèrent ne reçoivent aucun label de groupe : chacune contribue avec les labels de ses deux images.

## Méthode

- Source unique : `analysis/output/final_training_manifest.csv` ; cible : `effective_label`.
- 500 partitions candidates testées à partir du seed global `20260902`.
- Pour chaque candidat, les groupes sont mélangés, puis coupés au plus près de 70/15/15 en nombre d’images sans casser de groupe.
- Score : `100 × taille + 10 × classes + 5 × années + 2 × caméras + 1 × positions`, chaque composante étant le maximum absolu des écarts en points de pourcentage.
- Candidat retenu : index `191`, seed `20948826` ; score initial `6.764761`, puis `0.469407` après 137 échanges acceptés sur 50000 essais. Les échanges portent uniquement sur des groupes de même taille afin de conserver exactement les volumes.

## Résultats

### Classes

| classe | train | validation | test |
| --- | --- | --- | --- |
| Conforme | 7582 (30.73 %) | 1625 (30.72 %) | 1625 (30.73 %) |
| NON Conforme | 3964 (16.06 %) | 850 (16.07 %) | 850 (16.07 %) |
| PIETRA | 4224 (17.12 %) | 905 (17.11 %) | 905 (17.11 %) |
| Vide | 8907 (36.09 %) | 1909 (36.09 %) | 1908 (36.08 %) |

Le test contient exactement 1625 Conforme, 850 NON Conforme et 905 PIETRA. Ces volumes permettront d’interpréter au §4.2 les seuils métier de rappel ≥ 85 % et de précision ≥ 95 % sur Conforme ; aucune métrique modèle n’est calculée ici.

### Années

| année | train | validation | test |
| --- | --- | --- | --- |
| 2025 | 14186 (57.49 %) | 3040 (57.48 %) | 3040 (57.49 %) |
| 2026 | 10491 (42.51 %) | 2249 (42.52 %) | 2248 (42.51 %) |

### Positions

| position | train | validation | test |
| --- | --- | --- | --- |
| T | 11948 (48.42 %) | 2559 (48.38 %) | 2558 (48.37 %) |
| B | 12729 (51.58 %) | 2730 (51.62 %) | 2730 (51.63 %) |

### Caméras

| caméra | train | validation | test |
| --- | --- | --- | --- |
| 1 | 4068 (16.48 %) | 871 (16.47 %) | 872 (16.49 %) |
| 2 | 4166 (16.88 %) | 894 (16.90 %) | 893 (16.89 %) |
| 3 | 4310 (17.47 %) | 924 (17.47 %) | 923 (17.45 %) |
| 4 | 4055 (16.43 %) | 868 (16.41 %) | 869 (16.43 %) |
| 5 | 4139 (16.77 %) | 887 (16.77 %) | 887 (16.77 %) |
| 6 | 3939 (15.96 %) | 845 (15.98 %) | 844 (15.96 %) |

### Cas difficiles

| flag | train | validation | test |
| --- | --- | --- | --- |
| multiple | 851 (3.45 %) | 174 (3.29 %) | 172 (3.25 %) |
| chunk | 3513 (14.24 %) | 738 (13.95 %) | 724 (13.69 %) |
| mixed_quality | 0 (0.00 %) | 0 (0.00 %) | 0 (0.00 %) |

Les flags difficiles ne constituent pas une contrainte forte du score ; leur distribution est contrôlée après coup afin de ne pas complexifier artificiellement l’optimisation. L’écart maximal au global est de 0.4205 point : aucun déséquilibre fort n’est observé sur `multiple`, `chunk` ou `mixed_quality`.

### Paires historiques

| indicateur | train | validation | test |
| --- | --- | --- | --- |
| paired_images | 22688 | 4824 | 4840 |
| unpaired_images | 1989 | 465 | 448 |
| complete_pairs | 11344 | 2412 | 2420 |

### Écarts au global

| dimension | train max abs pp | validation max abs pp | test max abs pp | maximum |
| --- | --- | --- | --- | --- |
| effective_label | 0.0027 | 0.0048 | 0.0107 | 0.0107 |
| year | 0.0011 | 0.0079 | 0.003 | 0.0079 |
| cam_num | 0.0039 | 0.0179 | 0.0101 | 0.0179 |
| cam_position | 0.0117 | 0.0224 | 0.0322 | 0.0322 |

Les valeurs détaillées catégorie par catégorie sont disponibles dans `ml_split_summary.json` et `ml_split_comparison.csv`.

### Tableau global de comparaison

| indicateur | Global | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| Volume images | 35 254 | 24 677 | 5 289 | 5 288 |
| Conforme | 30.73 % | 30.73 % | 30.72 % | 30.73 % |
| NON Conforme | 16.07 % | 16.06 % | 16.07 % | 16.07 % |
| PIETRA | 17.12 % | 17.12 % | 17.11 % | 17.11 % |
| Vide | 36.09 % | 36.09 % | 36.09 % | 36.08 % |
| 2025 | 57.49 % | 57.49 % | 57.48 % | 57.49 % |
| 2026 | 42.51 % | 42.51 % | 42.52 % | 42.51 % |
| Top | 48.41 % | 48.42 % | 48.38 % | 48.37 % |
| Bottom | 51.59 % | 51.58 % | 51.62 % | 51.63 % |
| Caméra 1 | 16.48 % | 16.48 % | 16.47 % | 16.49 % |
| Caméra 2 | 16.89 % | 16.88 % | 16.90 % | 16.89 % |
| Caméra 3 | 17.46 % | 17.47 % | 17.47 % | 17.45 % |
| Caméra 4 | 16.43 % | 16.43 % | 16.41 % | 16.43 % |
| Caméra 5 | 16.77 % | 16.77 % | 16.77 % | 16.77 % |
| Caméra 6 | 15.96 % | 15.96 % | 15.98 % | 15.96 % |

## Satisfaisant

- aucun filename, `split_group_id` ou `historical_pair_id` ne traverse deux splits ;
- les volumes sont au plus près de 70/15/15 sans casser les paires ;
- chaque split contient les quatre classes, les deux années, Top et Bottom, ainsi que les six caméras ;
- le test contient plusieurs centaines d’images dans chacune des classes métier ;
- l’affectation est déterministe et reproductible.

## Limites

- les paires historiques demeurent heuristiques et ne prouvent pas l’identité physique ;
- l’équilibrage reproduit le dataset fourni, pas nécessairement le monde réel ;
- l’hétérogénéité 2025/2026 est répartie mais non supprimée ;
- la sélection du meilleur parmi plusieurs candidats réduit les écarts observés sans garantir un optimum mathématique global ;
- `multiple` et `chunk` sont seulement audités après affectation.

## Contrôles anti-fuite

| contrôle | résultat |
| --- | --- |
| filename_unique_and_in_one_split | PASS |
| split_group_id_in_one_split | PASS |
| historical_pair_id_in_one_split | PASS |
| train_inter_validation_empty | PASS |
| train_inter_test_empty | PASS |
| validation_inter_test_empty | PASS |
| union_35254_images | PASS |
| all_images_have_split | PASS |
| only_three_expected_splits | PASS |
| four_classes_in_each_split | PASS |
| both_years_in_each_split | PASS |
| top_bottom_in_each_split | PASS |
| cameras_1_to_6_in_each_split | PASS |

## Figures

- `analysis/output/split_figures/classes_by_split.png`
- `analysis/output/split_figures/years_by_split.png`
- `analysis/output/split_figures/cameras_by_split.png`
- `analysis/output/split_figures/positions_by_split.png`

## Décision

Les splits sont suffisamment proches du dataset global pour commencer les entraînements : l’écart maximal observé sur les dimensions prioritaires est de 0.0322 point de pourcentage, tous les contrôles anti-fuite passent et chaque sous-ensemble contient les quatre classes, les deux années, les deux positions et les six caméras.

Le jeu de test est désormais figé. Aucun entraînement, DataLoader, augmentation, suivi MLflow ou calcul de métrique modèle n’a été lancé.
