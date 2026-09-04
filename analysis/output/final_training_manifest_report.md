# Manifeste final d'entraînement — §4.1

## Dataset final

Le manifeste final contient 35254 images, toutes conservées. La distribution est calculée sur `effective_label` :

| effective_label | count | percentage |
| --- | --- | --- |
| Vide | 12724 | 36.09 |
| Conforme | 10832 | 30.73 |
| PIETRA | 6034 | 17.12 |
| NON Conforme | 5664 | 16.07 |

## Construction de effective_label

- Images hors des 59 conflits : reprise de `label_principal` (`principal`).
- `KEEP_PRINCIPAL` : reprise de `label_principal`, confirmée par la revue humaine.
- `CORRECT_TO_VIDE` : proposition dérivée `Vide` issue de la revue binaire.
- Cas NonVide ambigus : décision déterminée de la seconde revue multiclasses.

Les sources `labels_principal.csv` et `labels_masked.csv` restent intactes. `label_principal` est conservé à côté de `effective_label` pour assurer la traçabilité.

## Revue humaine

35 254 images → 59 conflits détectés → revue binaire en aveugle → 9 revues multiclasses en aveugle → 0 cas indéterminé. La revue multiclasses a produit 3 Conforme, 5 NON Conforme et 1 PIETRA.

Repères opérationnels utilisés pendant la revue humaine — et non définitions réglementaires officielles :

- **Conforme** : fruit clair, sain visuellement, sans moisissure ni défaut important.
- **NON Conforme** : fruit très noir, fortement dégradé, éclaté ou présentant des défauts importants.
- **PIETRA** : état intermédiaire entre Conforme et NON Conforme.

## Choix

Toutes les images sont conservées. Seuls les 16 cas explicitement corrigés par les revues humaines reçoivent un `effective_label` différent de `label_principal`.

- `multiple` : conservé.
- `chunk` : conservé.
- `mixed_quality` : conservé, même si aucune valeur True n'est présente actuellement.
- `missing labeled_by` : conservé ; l'absence de provenance est un problème de traçabilité, pas une preuve d'erreur.

Les simulations ont montré que retirer `chunk`, `multiple` ou les provenances manquantes introduisait des biais importants.

## Transitions de labels

| original_label | effective_label | count | is_correction |
| --- | --- | --- | --- |
| Conforme | Vide | 7 | True |
| Vide | Conforme | 3 | True |
| Vide | NON Conforme | 5 | True |
| Vide | PIETRA | 1 | True |
| Conforme | Conforme | 10829 | False |
| NON Conforme | NON Conforme | 5659 | False |
| PIETRA | PIETRA | 6033 | False |
| Vide | Vide | 12717 | False |

## Statistiques complémentaires

Années :

| year | count | percentage |
| --- | --- | --- |
| 2025 | 20266 | 57.49 |
| 2026 | 14988 | 42.51 |

Positions :

| position | count | percentage |
| --- | --- | --- |
| B | 18189 | 51.59 |
| T | 17065 | 48.41 |

Caméras :

| camera | count | percentage |
| --- | --- | --- |
| 3 | 6157 | 17.46 |
| 2 | 5953 | 16.89 |
| 5 | 5913 | 16.77 |
| 1 | 5811 | 16.48 |
| 4 | 5792 | 16.43 |
| 6 | 5628 | 15.96 |

Provenance de `effective_label` :

| source | count | percentage |
| --- | --- | --- |
| principal | 35195 | 99.83 |
| principal_confirmed_by_human | 43 | 0.12 |
| human_multiclass_review | 9 | 0.03 |
| human_binary_review | 7 | 0.02 |

## Appariement T/B historique et futur split

32352 images appartiennent à 16176 paires historiques heuristiques et 2902 restent non appariées. Parmi les paires, 10721 ont le même label effectif et 5455 des labels différents.

`split_group_id` prépare le futur split sans le créer : les deux vues d'une paire partagent le même groupe ; chaque image non appariée possède un groupe déterministe propre. L'appariement historique reste heuristique et ne constitue pas une vérité physique certaine.

## Données vidéo

Les 52 crops de `analysis/extracted_video/` ne sont pas intégrés au manifeste ML historique. Ils n'ont pas nécessairement suivi le même protocole de labellisation quatre classes ; le mot « conforme » dans le nom des AVI ne constitue pas un label.

## Implication

Le dataset final conserve les situations difficiles observées en production tout en appliquant uniquement des corrections humaines ciblées et traçables.

## Point satisfaisant

- 100 % des images conservées ;
- corrections ciblées ;
- traçabilité complète dans les artefacts dérivés ;
- sources originales intactes.

## Critique

- certaines corrections reposent sur une revue humaine subjective ;
- l'appariement historique T/B est heuristique ;
- la provenance `labeled_by` reste incomplète ;
- 2025 et 2026 sont hétérogènes ;
- `mixed_quality` n'est actuellement pas utilisé ;
- `chunk` et `multiple` rendent le problème plus difficile, mais plus réaliste.

## Lien avec le cahier des charges

Pour préparer le §4.2, la classe Conforme devra atteindre un rappel ≥ 85 % et une précision ≥ 95 %. Une erreur NON Conforme / PIETRA → Conforme est plus critique métier qu'un Conforme → rejet. Aucun label, poids de classe, split ou modèle n'est modifié ici.

## Validations techniques

| validation | result |
| --- | --- |
| row_count_35254 | PASS |
| filename_unique_35254 | PASS |
| effective_label_populated_35254 | PASS |
| exactly_four_effective_labels | PASS |
| effective_label_exact_spelling | PASS |
| no_image_lost | PASS |
| no_image_duplicated | PASS |
| no_pending_adjudication | PASS |
| no_uncertain_adjudication | PASS |
| effective_label_sources_valid | PASS |
| adjudication_statuses_valid | PASS |
| historical_paired_images_32352 | PASS |
| historical_unpaired_images_2902 | PASS |
| historical_pair_ids_16176 | PASS |
| each_pair_has_two_members | PASS |
| pair_members_share_split_group_id | PASS |
| no_pair_id_shared_between_pairs | PASS |
| each_unpaired_image_has_own_split_group_id | PASS |
| all_split_groups_are_disjoint | PASS |
