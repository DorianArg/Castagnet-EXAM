# Simulation des stratégies de constitution du dataset final

Cette analyse ne supprime aucune image et ne crée ni dataset final ni split train/validation/test. `label_principal` reste la cible. Les désaccords T/B et les changements depuis `label_filename` sont mesurés, jamais utilisés comme filtres.

## Vue d'ensemble

| strategy | kept | removed | kept_pct | complete_pairs | partial_pairs |
| --- | --- | --- | --- | --- | --- |
| S0 | 35254 | 0 | 100.0 | 16176 | 0 |
| S1 | 35195 | 59 | 99.83 | 16131 | 45 |
| S2 | 29043 | 6211 | 82.38 | 11229 | 4245 |
| S3a | 29043 | 6211 | 82.38 | 11229 | 4245 |
| S3b | 22103 | 13151 | 62.7 | 6600 | 6795 |

S2 et S3a sont identiques parce que les critères fournis sont identiques. Cette redondance est conservée explicitement plutôt que d'inventer une règle supplémentaire.

## Revue humaine des conflits masked/principal

59 conflits ont été détectés sur 35 254 images et revus manuellement en aveugle. La première revue portait uniquement sur `Vide` / `NonVide` : 45 décisions Vide, 14 NonVide et 0 Incertain. Elle confirme directement 43 labels principaux et propose 7 corrections dérivées vers Vide.

Une décision NonVide ne suffit pas à reconstruire une classe parmi Conforme, NON Conforme et PIETRA. Une seconde revue multiclasses est donc nécessaire uniquement pour les 9 cas concernés. Aucune source existante — `label_principal`, `labels_masked`, `label_filename` ou la vue T/B associée — n'est privilégiée automatiquement.

`S1_REVIEWED` réintègre les 59 cas et conserve 35254 images. Les 9 décisions multiclasses sont terminées : aucun cas pending ou uncertain ne subsiste. Les labels effectifs restent dérivés et ne sont pas écrits dans les CSV sources.

## Pourquoi une revue manuelle ciblée ?

Le volume de 59 cas est faible. La revue humaine évite une exclusion automatique inutile, une confiance arbitraire dans `labels_masked` ou une confiance arbitraire dans `label_principal`. Elle reste subjective et les décisions `Incertain` sont conservées comme telles et exclues tant qu'aucune adjudication fiable n'existe.


## Flags atomiques

| flag | images |
| --- | --- |
| is_masked_disagreement | 59 |
| is_multiple | 1197 |
| is_chunk | 4975 |
| is_mixed_quality | 0 |
| is_missing_labeled_by | 8033 |
| is_label_filename_changed | 12726 |
| is_historical_paired | 32352 |
| is_historical_unpaired | 2902 |
| has_pair_label_disagreement | 10924 |

## Impact isolé de chaque filtre

| filter | images_removed_alone |
| --- | --- |
| is_masked_disagreement | 59 |
| is_multiple | 1197 |
| is_chunk | 4975 |
| is_mixed_quality | 0 |
| is_missing_labeled_by | 8033 |

Les répartitions par classe, année, position et caméra de chaque filtre sont conservées dans le résumé JSON.

## Chevauchements principaux

| intersection | images |
| --- | --- |
| multiple_and_chunk | 18 |
| chunk_and_masked_disagreement | 2 |
| multiple_and_masked_disagreement | 0 |
| missing_labeled_by_and_vide | 3000 |
| missing_labeled_by_and_multiple | 388 |
| missing_labeled_by_and_chunk | 711 |
| missing_labeled_by_and_masked_disagreement | 5 |
| multiple_and_chunk_and_masked_disagreement | 0 |
| missing_labeled_by_and_multiple_and_chunk | 11 |
| union_masked_multiple_chunk_mixed_quality | 6211 |
| missing_labeled_by_and_any_quality_filter | 1093 |

Les retraits des stratégies sont calculés sur l'union réelle des filenames et non par addition de ces compteurs.

## S0 — Baseline large

### Choix

Conserver toutes les images dont label_principal est valide.

### Hypothèse

Faire confiance à la cible d'entraînement actuellement validée.

### Conséquence data

- 35254 images conservées (100.00 %), 0 retirées.
- 16176 paires historiques complètes, 0 paires partielles et 0 entièrement retirées.
- Parmi les paires complètes : 10714 labels principaux identiques et 5462 différents.
- Écart absolu maximal de proportion par rapport à S0 : 0.00 point(s).

Classes :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| Conforme | 10836 | 30.74 | 0.0 |
| NON Conforme | 5659 | 16.05 | 0.0 |
| PIETRA | 6033 | 17.11 | 0.0 |
| Vide | 12726 | 36.1 | 0.0 |

Années :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 2025 | 20266 | 57.49 | 0.0 |
| 2026 | 14988 | 42.51 | 0.0 |

Positions :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| B | 18189 | 51.59 | 0.0 |
| T | 17065 | 48.41 | 0.0 |

Caméras :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 1 | 5811 | 16.48 | 0.0 |
| 2 | 5953 | 16.89 | 0.0 |
| 3 | 6157 | 17.46 | 0.0 |
| 4 | 5792 | 16.43 | 0.0 |
| 5 | 5913 | 16.77 | 0.0 |
| 6 | 5628 | 15.96 | 0.0 |

### Point positif

Préserve toute la diversité observée en production.

### Critique

Conserve aussi les 59 conflits masked qui méritent une revue humaine.
## S1 — Prudente

### Choix

Exclure provisoirement les seuls conflits masked vide/non-vide.

### Hypothèse

Isoler un petit ensemble contradictoire avant revue humaine.

### Conséquence data

- 35195 images conservées (99.83 %), 59 retirées.
- 16131 paires historiques complètes, 45 paires partielles et 0 entièrement retirées.
- Parmi les paires complètes : 10706 labels principaux identiques et 5425 différents.
- Écart absolu maximal de proportion par rapport à S0 : 0.08 point(s).

Classes :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| Conforme | 10824 | 30.75 | 0.01 |
| NON Conforme | 5659 | 16.08 | 0.03 |
| PIETRA | 6033 | 17.14 | 0.03 |
| Vide | 12679 | 36.03 | -0.07 |

Années :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 2025 | 20207 | 57.41 | -0.08 |
| 2026 | 14988 | 42.59 | 0.08 |

Positions :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| B | 18168 | 51.62 | 0.03 |
| T | 17027 | 48.38 | -0.03 |

Caméras :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 1 | 5805 | 16.49 | 0.01 |
| 2 | 5949 | 16.9 | 0.01 |
| 3 | 6146 | 17.46 | 0.0 |
| 4 | 5773 | 16.4 | -0.03 |
| 5 | 5906 | 16.78 | 0.01 |
| 6 | 5616 | 15.96 | 0.0 |

### Point positif

Réduit l'incertitude explicite avec une perte très limitée.

### Critique

Le masque automatique n'est pas une vérité terrain et peut lui-même être erroné.
## S2 — Contrôlée

### Choix

Conserver seulement les cas sans conflit masked, multiple, chunk ou qualité mixte.

### Hypothèse

Mesurer un dataset centré sur des images simples.

### Conséquence data

- 29043 images conservées (82.38 %), 6211 retirées.
- 11229 paires historiques complètes, 4245 paires partielles et 702 entièrement retirées.
- Parmi les paires complètes : 7348 labels principaux identiques et 3881 différents.
- Écart absolu maximal de proportion par rapport à S0 : 7.56 point(s).

Classes :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| Conforme | 7922 | 27.28 | -3.46 |
| NON Conforme | 4667 | 16.07 | 0.02 |
| PIETRA | 3775 | 13.0 | -4.11 |
| Vide | 12679 | 43.66 | 7.56 |

Années :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 2025 | 15873 | 54.65 | -2.84 |
| 2026 | 13170 | 45.35 | 2.84 |

Positions :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| B | 14878 | 51.23 | -0.36 |
| T | 14165 | 48.77 | 0.36 |

Caméras :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 1 | 5177 | 17.83 | 1.35 |
| 2 | 4998 | 17.21 | 0.32 |
| 3 | 4811 | 16.57 | -0.89 |
| 4 | 4559 | 15.7 | -0.73 |
| 5 | 4929 | 16.97 | 0.2 |
| 6 | 4569 | 15.73 | -0.23 |

### Point positif

Simplifie le signal visuel et l'analyse initiale.

### Critique

Peut retirer des situations normales de production et déplacer fortement les distributions.
## S3a — Stricte

### Choix

Appliquer les critères stricts sans exiger la provenance labeled_by.

### Hypothèse

Distinguer la qualité visuelle de la traçabilité d'annotation.

### Conséquence data

- 29043 images conservées (82.38 %), 6211 retirées.
- 11229 paires historiques complètes, 4245 paires partielles et 702 entièrement retirées.
- Parmi les paires complètes : 7348 labels principaux identiques et 3881 différents.
- Écart absolu maximal de proportion par rapport à S0 : 7.56 point(s).

Classes :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| Conforme | 7922 | 27.28 | -3.46 |
| NON Conforme | 4667 | 16.07 | 0.02 |
| PIETRA | 3775 | 13.0 | -4.11 |
| Vide | 12679 | 43.66 | 7.56 |

Années :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 2025 | 15873 | 54.65 | -2.84 |
| 2026 | 13170 | 45.35 | 2.84 |

Positions :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| B | 14878 | 51.23 | -0.36 |
| T | 14165 | 48.77 | 0.36 |

Caméras :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 1 | 5177 | 17.83 | 1.35 |
| 2 | 4998 | 17.21 | 0.32 |
| 3 | 4811 | 16.57 | -0.89 |
| 4 | 4559 | 15.7 | -0.73 |
| 5 | 4929 | 16.97 | 0.2 |
| 6 | 4569 | 15.73 | -0.23 |

### Point positif

Ne confond pas provenance manquante et label faux.

### Critique

Avec les règles fournies, cette stratégie est exactement identique à S2.
## S3b — Stricte avec provenance

### Choix

Ajouter aux critères stricts l'obligation d'un labeled_by renseigné.

### Hypothèse

Mesurer l'effet d'une exigence maximale de traçabilité.

### Conséquence data

- 22103 images conservées (62.70 %), 13151 retirées.
- 6600 paires historiques complètes, 6795 paires partielles et 2781 entièrement retirées.
- Parmi les paires complètes : 4177 labels principaux identiques et 2423 différents.
- Écart absolu maximal de proportion par rapport à S0 : 7.70 point(s).

Classes :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| Conforme | 6723 | 30.42 | -0.32 |
| NON Conforme | 2553 | 11.55 | -4.5 |
| PIETRA | 3145 | 14.23 | -2.88 |
| Vide | 9682 | 43.8 | 7.7 |

Années :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 2025 | 11022 | 49.87 | -7.62 |
| 2026 | 11081 | 50.13 | 7.62 |

Positions :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| B | 12759 | 57.73 | 6.14 |
| T | 9344 | 42.27 | -6.14 |

Caméras :

| category | count | pct | delta_vs_S0_pp |
| --- | --- | --- | --- |
| 1 | 3192 | 14.44 | -2.04 |
| 2 | 3011 | 13.62 | -3.27 |
| 3 | 3770 | 17.06 | -0.4 |
| 4 | 3898 | 17.64 | 1.21 |
| 5 | 4552 | 20.59 | 3.82 |
| 6 | 3680 | 16.65 | 0.69 |

### Point positif

Toutes les images retenues possèdent une provenance explicite.

### Critique

L'absence de pseudo ne prouve aucune erreur et peut introduire un biais massif.


## Analyse des changements vers Vide

Pour chaque stratégie, le JSON conserve la répartition des images `label_filename != label_principal` devenues `Vide` par année, caméra, position, labeliseur canonique et couverture masked. Ces images ne sont jamais exclues sur ce seul motif.

## Lecture quantitative

S1 constitue le compromis quantitatif le plus conservateur : elle isole uniquement les 59 conflits masked connus tout en conservant les cas `multiple`, `chunk`, `Vide`, les provenances manquantes et les désaccords entre vues. Cette observation ne vaut pas validation métier. S2/S3a mesurent le coût d'un dataset artificiellement simplifié. S3b mesure surtout une exigence de traçabilité et ne doit pas être interprétée comme une amélioration certaine des labels.

## Vigilance pour le futur split

Le split n'est pas créé ici. Lorsqu'il le sera, les deux vues d'une même paire historique devront probablement rester dans le même split afin de limiter les fuites de données.
