# Rapport qualité du dataset

## Vue d'ensemble

- Images : 35254
- Filenames uniques : 35254
- Images avec `batch_num` : 9986

## Distribution globale de label_principal

| label | count | percentage |
| --- | --- | --- |
| Vide | 12726 | 36.1 |
| Conforme | 10836 | 30.74 |
| PIETRA | 6033 | 17.11 |
| NON Conforme | 5659 | 16.05 |

Les distributions par année, caméra et position sont disponibles dans
`label_principal_distribution.csv`.

## Distribution globale de label_filename

| label | count | percentage |
| --- | --- | --- |
| Conforme | 14818 | 42.03 |
| PIETRA | 11267 | 31.96 |
| NON Conforme | 9169 | 26.01 |

Les distributions par année, caméra et position sont disponibles dans
`label_filename_distribution.csv`.

## Relecture

| reviewed | count | percentage |
| --- | --- | --- |
| True | 35254 | 100.0 |
| False | 0 | 0.0 |

- Images relues sans `labeled_by` : 8033

## Identifiants des labeliseurs

Les identifiants `NicoG`, `Nico`, `nico` et `nico h` correspondent au même
annotateur. Ils sont regroupés sous `Nico` uniquement pour les statistiques ;
les valeurs brutes restent conservées dans le manifeste.

- Identifiants non vides avant normalisation : 7
- Labeliseurs après normalisation : 4
- Images attribuées à `Nico` après regroupement : 15518

La distribution brute est disponible dans `labeled_by_raw_distribution.csv`
et la distribution canonisée dans `labeled_by_distribution.csv`.

## Cas ambigus

| indicateur | nombre |
| --- | ---: |
| multiple | 1197 |
| chunk | 4975 |
| mixed_quality | 0 |
| label_filename différent de label_principal | 12726 |
| labels_masked en désaccord vide/non-vide avec label_principal | 59 |
| images uniques avec au moins un indicateur | 18890 |

## labels_masked

- Images couvertes : 6577 (18.66 %)
- Désaccords vide/non-vide avec `label_principal` : 59

| masked_label | count | percentage_of_covered |
| --- | --- | --- |
| chataigne | 4190 | 63.71 |
| vide | 1572 | 23.9 |
| chunks | 586 | 8.91 |
| multiple | 229 | 3.48 |

`masked_label` est conservé comme information distincte et ne remplace jamais
`label_principal`.
