# Notes d'extraction manuelle des vidéos

## Caractéristiques observées

| Flux | Durée | Frames | FPS | Résolution | Codec FourCC |
| --- | ---: | ---: | ---: | ---: | --- |
| Bottom (`extrait-cam-1-b-conforme.avi`) | 30,000 s | 750 | 25 | 720 × 576 | `dvsd` |
| Top (`extrait-cam-1-t-conforme.avi`) | 30,000 s | 750 | 25 | 720 × 576 | `dvsd` |

Les 750 frames de chaque fichier ont été décodées lors de l'inspection. La
première frame est horodatée à 0 ms et la dernière commence à 29 960 ms.

## Choix effectués

- L'extraction et la définition des crops sont manuelles, conformément à la
  consigne du sujet.
- L'opérateur parcourt les frames et ne retient que celles où il décide qu'une
  châtaigne doit être extraite.
- Le crop est un rectangle dessiné manuellement autour de la châtaigne avec
  l'outil ROI d'OpenCV.
- Une frame contenant plusieurs châtaignes peut produire plusieurs crops :
  l'opérateur relance la sélection sur la même frame pour chaque fruit retenu.
- Les indices de frame commencent à 0. Le timestamp enregistré est calculé à
  partir de cet indice et du FPS déclaré par la vidéo (25 FPS).
- La frame source, sans le bandeau d'aide affiché par l'interface, est enregistrée
  une seule fois dès que son premier crop est validé.
- Le mot `conforme` présent dans le nom des vidéos n'est pas converti en label
  de qualité pour les crops.

### Justification de la méthode manuelle

Le sujet demande explicitement un découpage et un crop manuels. Les deux
extraits de 30 secondes sont suffisamment courts pour permettre une vérification
humaine. La faible résolution et le contraste font partie des conditions
réelles d'acquisition ; une détection automatique ajouterait une source d'erreur
avant la constitution du sous-ensemble de référence. L'approche manuelle produit
un sous-ensemble contrôlé et traçable qui pourra ensuite servir à l'analyse ou à
l'évaluation de méthodes automatisées. Elle n'est toutefois pas parfaite : le
choix des frames et le cadrage restent subjectifs et dépendent de l'opérateur.

## Hypothèses

- Les deux flux représentent le fonctionnement de la même machine, comme
  indiqué par le matériel fourni.
- Aucune synchronisation Top/Bottom n'est supposée à ce stade, malgré des
  caractéristiques vidéo identiques.
- Une frame non retenue signifie seulement qu'aucun crop manuel n'a été créé ;
  elle ne reçoit pas automatiquement un diagnostic `Vide`.

## Implications sur les données

- La sélection humaine n'est pas nécessairement exhaustive.
- Le choix des frames peut introduire un biais de sélection.
- Les dimensions et les marges des crops peuvent varier selon l'opérateur.
- Ce petit sous-ensemble n'est pas nécessairement représentatif des 35 254
  images du dataset principal.
- Les crops ne disposent pas encore d'un label de qualité ni d'une
  correspondance entre les vues T et B.

## Points satisfaisants

- La relation entre vidéo, position caméra, frame, timestamp et crop est
  conservée explicitement dans `crops_metadata.csv`.
- Plusieurs châtaignes peuvent être documentées séparément sur une même frame.
- Les vidéos, images et CSV sources ne sont pas modifiés.
- Le contrôle humain est adapté à des extraits courts et à l'objectif demandé.

## Limites / critiques

- La méthode manuelle est peu scalable et peut être fatigante sur 1 500 frames.
- Le cadrage peut varier entre opérateurs ou entre deux sessions.
- L'outil ne garantit pas que toutes les châtaignes visibles ont été retenues.
- Les deux flux ont la même durée, le même FPS et le même nombre de frames, mais
  cela ne prouve pas qu'ils sont synchronisés.
- L'extraction manuelle seule ne crée aucune correspondance T/B ; l'alignement
  temporel dérivé est documenté séparément ci-dessous.

## Correspondance Top / Bottom

### Choix

La ressemblance visuelle n'est pas utilisée pour apparier les vues, car les
caméras Top et Bottom montrent deux faces différentes du fruit. L'appariement
repose sur la dynamique temporelle du passage dans la machine. Les séquences
sont triées par `frame_index`, puis par `crop_index` uniquement pour assurer un
ordre stable ; le rang et le numéro du crop ne servent jamais d'identifiant de
correspondance.

L'alignement monotone maximise d'abord le nombre de paires admissibles, puis
minimise leur coût temporel total. Il autorise des gaps dans les deux séquences.

### Fonction de coût

`cost(T, B) = abs((frame_B - frame_T) - 7)`

Seuls les deltas +6, +7 et +8 sont admissibles, soit un coût inférieur ou égal
à 1.

### Observation

La comparaison manuelle initiale montre un décalage très stable de +6 à +7
frames entre la vue Top et la vue Bottom, majoritairement +7 frames. À 25 FPS,
ces décalages correspondent respectivement à environ 240 et 280 ms.

### Hypothèse

Une même châtaigne apparaît environ 7 frames plus tard sur le flux Bottom que
sur le flux Top.

### Implication

Les résultats `matched_temporal` sont des correspondances inférées à partir du
temps, et non une vérité terrain directement observable. Les gaps permettent de
laisser une vue sans correspondant sans décaler les associations suivantes.

### Point satisfaisant

La stratégie tolère les vues manquantes sans créer de décalage cumulatif. Elle
teste également si chaque paire appartient à toutes les solutions optimales :
une association non forcée est conservée comme `ambiguous` au lieu d'être
choisie arbitrairement.

### Cas ambigu réel

Deux crops Top existent à la frame 572 :

- `T_frame_000572_crop_01.jpg`
- `T_frame_000572_crop_02.jpg`

pour un seul crop Bottom à la frame 579 :

- `B_frame_000579_crop_01.jpg`

Les deux associations ont un delta de +7 et un coût nul. La temporalité seule
ne permet pas de les départager ; les trois crops sont donc conservés dans un
groupe `ambiguous` et aucune des deux paires n'est déclarée certaine.

### Limites / critique

- Certaines châtaignes ne sont visibles que sur une vue.
- Plusieurs fruits peuvent apparaître dans une même fenêtre temporelle.
- Les deux faces d'une même châtaigne peuvent être très différentes.
- Une relation temporelle stable n'est pas une preuve absolue d'identité.
- L'observation porte uniquement sur un extrait de 30 secondes.
- Un artefact technique provenant d'anciennes sessions avait laissé 56 lignes
  de métadonnées orphelines. Après vérification que les fichiers correspondants
  n'existaient plus, ces lignes ont été exclues de `crops_metadata.csv` et
  conservées dans
  `archive/orphaned_crop_metadata_OBSOLETE_DO_NOT_USE.csv` pour traçabilité.
- Les 12 décisions issues de l'ancienne interface de comparaison visuelle sont
  archivées dans `archive/tb_pairs_visual_OBSOLETE_DO_NOT_USE.csv`. Elles ne
  participent ni à l'alignement temporel ni aux statistiques actives.
