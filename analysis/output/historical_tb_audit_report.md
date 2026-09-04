# Audit exploratoire de l'appariement Top/Bottom historique

Ce rapport est un audit de métadonnées. Il ne constitue pas une vérité terrain. La table historique produite séparément ne contient que des correspondances qualifiées d'heuristiques.

## Hypothèses testées

- **A** : `year + batch_num + cam_num + sample_num`
- **B** : clé A + `label_filename`
- **C** : `year + cam_num + sample_num + label_filename` (batch ignoré)
- **D** : `year + cam_num + sample_num` (batch et label historique ignorés)

La valeur `<NO_BATCH>` est utilisée comme sentinelle stable quand `batch_num` est absent. `cam_position` sert uniquement à compter les côtés. `label_principal`, l'ordre des fichiers et l'offset vidéo +7 ne sont jamais utilisés dans les clés.

## Résultats

| scope | key_id | exact_1t1b_groups | paired_images_pct | top_only_groups | top_only_images_pct | bottom_only_groups | bottom_only_images_pct | ambiguous_groups | ambiguous_images_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | A | 5561 | 31.55 | 437 | 1.28 | 1737 | 4.93 | 4819 | 62.33 |
| all | B | 16176 | 91.77 | 889 | 2.52 | 2013 | 5.71 | 0 | 0.0 |
| all | C | 12318 | 69.88 | 687 | 1.95 | 719 | 2.26 | 2641 | 26.36 |
| all | D | 932 | 5.29 | 184 | 0.54 | 126 | 0.38 | 5499 | 93.87 |
| 2025 | A | 4667 | 46.06 | 275 | 1.41 | 1691 | 8.35 | 2344 | 44.3 |
| 2025 | B | 8950 | 88.33 | 496 | 2.45 | 1870 | 9.23 | 0 | 0.0 |
| 2025 | C | 5092 | 50.25 | 294 | 1.46 | 576 | 3.23 | 2641 | 45.86 |
| 2025 | D | 38 | 0.38 | 22 | 0.12 | 80 | 0.43 | 3024 | 99.16 |
| 2026 | A | 894 | 11.93 | 162 | 1.11 | 46 | 0.31 | 2475 | 86.71 |
| 2026 | B | 7226 | 96.42 | 393 | 2.62 | 143 | 0.95 | 0 | 0.0 |
| 2026 | C | 7226 | 96.42 | 393 | 2.62 | 143 | 0.95 | 0 | 0.0 |
| 2026 | D | 894 | 11.93 | 162 | 1.11 | 46 | 0.31 | 2475 | 86.71 |

Le taux représente les images appartenant à un groupe exactement 1 Top + 1 Bottom. Les catégories « plusieurs Top/Bottom » se recouvrent volontairement dans le CSV détaillé ; `ambiguous_groups` est leur union.

### Effet de `batch_num`

| scope | comparison | ambiguous_groups_without_batch | ambiguous_groups_with_batch | ambiguous_groups_resolved_by_batch_split | original_ambiguous_groups_still_ambiguous |
| --- | --- | --- | --- | --- | --- |
| all | D->A | 5499 | 4819 | 680 | 4819 |
| all | C->B | 2641 | 0 | 2641 | 0 |
| 2025 | D->A | 3024 | 2344 | 680 | 2344 |
| 2025 | C->B | 2641 | 0 | 2641 | 0 |
| 2026 | D->A | 2475 | 2475 | 0 | 2475 |
| 2026 | C->B | 0 | 0 | 0 | 0 |

`ambiguous_groups_resolved_by_batch_split` compte les groupes ambigus sans batch qui, une fois séparés par batch, ne contiennent plus aucun sous-groupe ambigu. Ce chiffre mesure une collision de métadonnées résolue, pas une paire physique confirmée.

### Effet de `label_filename`

| scope | comparison | structural_key | with_label_filename_key | exact_pairs_without_label | exact_pairs_with_label | ambiguous_groups_without_label | ambiguous_groups_with_label | structural_exact_pairs_split_by_different_label_filename |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | A->B | A | B | 5561 | 16176 | 4819 | 0 | 8 |
| all | D->C | D | C | 932 | 12318 | 5499 | 2641 | 18 |
| 2025 | A->B | A | B | 4667 | 8950 | 2344 | 0 | 7 |
| 2025 | D->C | D | C | 38 | 5092 | 3024 | 2641 | 17 |
| 2026 | A->B | A | B | 894 | 7226 | 2475 | 0 | 1 |
| 2026 | D->C | D | C | 894 | 7226 | 2475 | 0 | 1 |

L'ajout du diagnostic historique augmente fortement l'unicité apparente. Cela peut séparer de vraies collisions, mais aussi empêcher une association entre deux vues dont le diagnostic historique diffère. Il ne s'agit donc pas d'une preuve d'identité.

## Origine documentée de `sample_num`

La recherche dans le dépôt ne retrouve **aucun code de création ou de renommage des 35 254 images**, ni aucune règle explicite d'incrément ou de remise à zéro du compteur. Le README documente seulement le format `annee_label_Cam_{T|B}_{numCamera}_{numEchantillon}.jpg`. `labeling_tool/build_labels_csv.py` et `analysis/build_dataset_manifest.py` parsèrent des noms déjà existants ; ils n'attribuent pas `sample_num`. L'application de labellisation ne renomme pas les images. Le rapport PDF historique répète le format sans expliquer le compteur. Aucun historique Git exploitable n'est présent dans cette copie du dépôt.

La logique de namespace n'est donc pas démontrée par le code ou la documentation disponible ; elle doit être évaluée empiriquement.

## Vérification empirique du namespace

| variant | year | contexts | continuous_contexts | contexts_starting_at_zero | total_holes | maximum_holes_in_one_context |
| --- | --- | --- | --- | --- | --- | --- |
| with_label_filename | 2025 | 48 | 20 | 45 | 361 | 55 |
| with_label_filename | 2026 | 36 | 15 | 33 | 290 | 59 |
| without_label_filename | 2025 | 36 | 24 | 35 | 125 | 46 |
| without_label_filename | 2026 | 12 | 7 | 12 | 48 | 19 |

Les métriques détaillées pour chacun des contextes sont enregistrées dans `historical_tb_sample_num_analysis.csv`. Avec `label_filename`, 78 contextes sur 84 commencent à zéro. Sans `label_filename`, 47 contextes sur 48 commencent à zéro. La réunion de plusieurs labels bouche artificiellement certains trous : elle ne constitue donc pas un meilleur compteur unique.

### Réutilisation d'un même numéro entre labels

| scope | sample_slots | reused_between_labels | reuse_pct | images_in_reused_slots | images_pct |
| --- | --- | --- | --- | --- | --- |
| all | 12538 | 4827 | 38.5 | 21990 | 62.38 |
| 2025 | 8965 | 2351 | 26.22 | 8992 | 44.37 |
| 2026 | 3573 | 2476 | 69.3 | 12998 | 86.72 |

Distribution globale : {'1': 7711, '2': 3114, '3': 1713}. La réutilisation est particulièrement forte en 2026 et soutient l'hypothèse de compteurs locaux au label historique.

### Cohérence littérale des noms pour la clé B

Sur 16176 groupes B exactement 1T/1B, 16176 (100.00 %) ont des noms dont toutes les composantes sont identiques sauf `Cam_T` / `Cam_B`. Il y a 0 exception et 0 nom non conforme au format attendu.

### Huit groupes A séparés par `label_filename`

| filename_top | filename_bottom | year | batch_num | cam_num | sample_num | label_filename_top | label_filename_bottom | label_principal_top | label_principal_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025_NON_Conforme_Cam_T_1_380.jpg | 2025_PIETRA_Cam_B_1_380.jpg | 2025 | <NO_BATCH> | 1 | 380 | NON Conforme | PIETRA | Vide | PIETRA |
| 2025_NON_Conforme_Cam_T_1_386.jpg | 2025_PIETRA_Cam_B_1_386.jpg | 2025 | <NO_BATCH> | 1 | 386 | NON Conforme | PIETRA | Vide | PIETRA |
| 2025_PIETRA_Cam_T_3_19.jpg | 2025_NON_Conforme_Cam_B_3_19.jpg | 2025 | <NO_BATCH> | 3 | 19 | PIETRA | NON Conforme | Vide | Vide |
| 2025_NON_Conforme_Cam_T_6_16.jpg | 2025_PIETRA_Cam_B_6_16.jpg | 2025 | <NO_BATCH> | 6 | 16 | NON Conforme | PIETRA | Vide | Vide |
| 2025_NON_Conforme_Cam_T_6_213.jpg | 2025_PIETRA_Cam_B_6_213.jpg | 2025 | <NO_BATCH> | 6 | 213 | NON Conforme | PIETRA | Vide | Vide |
| 2025_PIETRA_Cam_T_6_331.jpg | 2025_NON_Conforme_Cam_B_6_331.jpg | 2025 | <NO_BATCH> | 6 | 331 | PIETRA | NON Conforme | Vide | Vide |
| 2025_PIETRA_Cam_T_6_382.jpg | 2025_NON_Conforme_Cam_B_6_382.jpg | 2025 | <NO_BATCH> | 6 | 382 | PIETRA | NON Conforme | Vide | Vide |
| 2026_PIETRA_Cam_T_6_401.jpg | 2026_Conforme_Cam_B_6_401.jpg | 2026 | <NO_BATCH> | 6 | 401 | PIETRA | Conforme | Vide | Vide |

Ces cas associent seulement un Top et un Bottom parce que leur numéro local se rencontre dans deux labels historiques différents. Leurs noms ne suivent pas le motif symétrique T/B de la clé B ; les considérer comme des couples serait donc une collision numérique plausible, pas une association démontrée.

### Diagnostic de la clé B, meilleure au seul critère numérique

- Paires potentielles : 16176 (91.77 % des images).
- Même `label_principal` : 10714.
- `label_principal` différent : 5462.
- Au moins un côté `multiple` : 1098 ; différence entre côtés : 1063.
- Au moins un côté `chunk` : 3969 ; différence entre côtés : 3450.
- Au moins un côté `Vide` : 8622 ; différence entre côtés : 5462.

Les transitions détaillées Top → Bottom sont conservées dans `historical_tb_label_transitions.csv`.

## Structure de `sample_num`

```json
{
  "2025": {
    "images": 20266,
    "sample_num_min": 0,
    "sample_num_max": 536,
    "distinct_sample_num": 537,
    "sample_values_repeated": 533,
    "maximum_images_sharing_one_sample_num": 48,
    "sample_values_used_by_multiple_cameras": 533,
    "sample_values_used_by_multiple_batches": 530,
    "sample_values_used_by_multiple_historical_labels": 517
  },
  "2026": {
    "images": 14988,
    "sample_num_min": 0,
    "sample_num_max": 612,
    "distinct_sample_num": 613,
    "sample_values_repeated": 605,
    "maximum_images_sharing_one_sample_num": 36,
    "sample_values_used_by_multiple_cameras": 605,
    "sample_values_used_by_multiple_batches": 0,
    "sample_values_used_by_multiple_historical_labels": 425
  }
}
```

Les répétitions, ruptures et redémarrages sont détaillés par année/caméra/batch dans `historical_tb_sample_num_analysis.csv`. Elles montrent que `sample_num` est un identifiant local et largement réutilisé, insuffisant sans contexte.

## Classement et choix envisageable

Classement retenu : **B — clé fortement soutenue mais restant heuristique**.

La répétition des compteurs entre labels, leurs nombreux redémarrages à zéro et la cohérence littérale de 100 % des groupes B soutiennent fortement l'idée que `label_filename` appartient au namespace historique de `sample_num`. Toutefois, aucun générateur ni document métier disponible ne confirme cette règle et aucun identifiant physique de châtaigne ne permet de valider les groupes.

Dans ce rôle, `label_filename` n'est **ni la cible, ni une caractéristique ML**. Il sert uniquement à interpréter le namespace technique historique lors de l'audit T/B. `label_principal` demeure la cible d'entraînement et n'est jamais utilisé pour construire les groupes.

## Méthode retenue

La clé `year + batch_num + cam_num + sample_num + label_filename` est retenue pour construire les correspondances historiques lorsque le groupe contient exactement un Top et un Bottom. Elle identifie 16 176 groupes, couvre 32 352 images (91,77 %) et leurs noms sont tous strictement symétriques, seule la composante `Cam_T` / `Cam_B` différant.

### Choix

`label_filename` est conservé uniquement comme composante du namespace historique nécessaire à l'interprétation de `sample_num`. Il ne remplace jamais `label_principal`, n'est pas une cible ML et ne doit pas devenir une caractéristique d'entrée du modèle.

### Hypothèse et arguments

Le compteur `sample_num` semble local au contexte historique incluant le label du filename. Cette hypothèse est soutenue par la forte réutilisation des numéros entre labels, la symétrie de 100 % des groupes retenus, l'effet mesurable de `batch_num` en 2025 et l'absence d'ambiguïtés avec la clé B.

### Implication

Les lignes de `historical_tb_pairs.csv` peuvent servir à organiser et analyser conjointement les deux vues. Leur méthode est `historical_namespace` et leur statut `heuristic_pair`; elles ne doivent pas être présentées comme une vérité terrain physique garantie. Elles restent distinctes de `analysis/extracted_video/tb_pairs.csv`, fondé sur l'alignement temporel des vidéos.

### Critique

Le générateur historique et un identifiant physique de châtaigne restent absents. Des fichiers peuvent manquer, les comportements diffèrent entre 2025 et 2026 et une collision rare peut subsister sans être observable dans les métadonnées.

## Implications et critique

- Conserver `batch_num` évite des collisions 2025 ; l'ignorer fusionne les segments `_1`/`_2`.
- Ajouter `label_filename` améliore l'unicité, mais risque une séparation artificielle selon le diagnostic historique.
- L'absence d'identifiant physique de châtaigne empêche de prouver qu'un groupe 1T/1B représente le même objet.
- Les structures 2025 et 2026 diffèrent ; une règle unique n'a pas la même performance selon l'année.
- Des labels principaux différents entre faces sont possibles et ne doivent pas invalider automatiquement un groupe.
- Les indicateurs `multiple`, `chunk` et `Vide` signalent des groupes potentiellement moins fiables.
- Aucun offset temporel n'est supposé : la règle +7 est propre aux deux vidéos déjà traitées.
