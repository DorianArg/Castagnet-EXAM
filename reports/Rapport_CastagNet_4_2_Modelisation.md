# CASTAGNET — §4.2 Modélisation et évaluation

**Rapport final — Modélisation, entraînement, comparaison et évaluation**

Mise en Situation Professionnelle — MSc BIHAR  
Compétences C28 et C31  
3 septembre 2026

## Synthèse décisionnelle

> **Décision —** ResNet18 est retenu comme candidat qualité du §4.2. Le checkpoint de l'epoch 13 et le seuil Conforme 0,55 ont été figés avant l'ouverture du test.

Le travail compare une baseline CNN construite depuis zéro à trois architectures pré-entraînées sur ImageNet. L'emploi du transfer learning améliore nettement la séparation des classes proches, en particulier PIETRA. Sur validation, ResNet18 atteint 88,15 % d'accuracy et 85,21 % de macro-F1, contre 77,88 % et 66,22 % pour la baseline retenue.

Le seuil 0,55 rend l'acceptation Conforme plus stricte. Sur le test final unique, il réduit le compteur métier de faux Conforme de 188 à 166, mais augmente les Conforme rejetées de 204 à 235.

> **Cahier des charges partiellement atteint —** la configuration finale atteint le rappel minimal (85,54 % pour un objectif de 85 %), mais pas la précision (89,16 % pour un objectif de 95 %). Aucune adaptation post-test n'est autorisée.

Cette sélection reste prédictive, non embarquée. ResNet18 est le plus lourd des trois modèles pré-entraînés évalués ; sa latence, sa mémoire, son export ONNX et sa capacité à traiter douze flux sur la cible GTX 1060 3 Go seront mesurés au §4.3.

## Sommaire

1. 4.2.1 Objectifs et contraintes métier
2. 4.2.2 Construction des jeux train, validation et test
3. 4.2.3 Prétraitement et augmentation
4. 4.2.4 Baseline CNN maison
5. 4.2.5 Analyse des deux versions du CNN
6. 4.2.6 Mise en place du transfer learning
7. 4.2.7 Modèles pré-entraînés étudiés
8. 4.2.8 Suivi des expériences avec MLflow
9. 4.2.9 Comparaison des architectures
10. 4.2.10 Sélection de ResNet18
11. 4.2.11 Calibration du seuil Conforme
12. 4.2.12 Évaluation finale sur le jeu de test
13. 4.2.13 Analyse du respect du cahier des charges
14. 4.2.14 Prise en compte des contraintes d'embarquement
15. 4.2.15 Limites, recul critique et pistes d'amélioration
16. 4.2.16 Conclusion et transition vers le §4.3

## 4.2.1 Objectifs et contraintes métier

La tâche consiste à classer chaque image dans l'une des quatre classes exclusives : Conforme, NON Conforme, PIETRA ou Vide. La cible d'apprentissage est exclusivement effective_label, avec le mapping Conforme=0, NON Conforme=1, PIETRA=2 et Vide=3. Le nom du fichier et le label historique ne déterminent jamais la classe.

Deux contraintes structurent l'évaluation : une précision de la classe Conforme d'au moins 95 % et un rappel Conforme d'au moins 85 %. La qualité du produit final est prioritaire sur le rendement : lorsqu'un arbitrage est nécessaire, il est préférable de rejeter davantage de fruits réellement conformes plutôt que de laisser passer des fruits NON Conforme ou PIETRA comme Conforme.

> **Lecture des métriques —** la précision Conforme mesure la fiabilité d'une acceptation ; le rappel Conforme mesure la part des vrais Conforme effectivement acceptés. Les deux objectifs doivent être satisfaits simultanément.

Le contexte de déploiement futur ajoute une contrainte de calcul : NVIDIA GTX 1060 3 Go, ancien processeur Intel i7, douze flux caméra et export ONNX. Le nombre de paramètres est donc suivi dès ce chapitre, mais il ne permet pas à lui seul de conclure sur le débit ou la latence. Cette validation est volontairement reportée au §4.3.

## 4.2.2 Construction des jeux train, validation et test

Le dataset final issu du §4.1 contient 35 254 images uniques. Il a été conservé en entier : réduire arbitrairement ce volume aurait supprimé de l'information utile sans démontrer que les confusions entre Conforme, PIETRA et NON Conforme provenaient des données plutôt que du modèle.

**Tableau 1 — Répartition group-aware des données**

| Jeu | Images | Part | Groupes | Paires T/B | Conforme | PIETRA |
|---|---|---|---|---|---|---|
| Train | 24 677 | 69,9977 % | 13 333 | 11 344 | 30,73 % | 17,12 % |
| Validation | 5 289 | 15,0026 % | 2 877 | 2 412 | 30,72 % | 17,11 % |
| Test | 5 288 | 14,9997 % | 2 868 | 2 420 | 30,73 % | 17,11 % |

*Source : analysis/output/ml_split_summary.json*

Le ratio 70/15/15 conserve une base d'apprentissage large tout en donnant à la validation et au test plus de cinq mille images chacun, quantité suffisante pour produire des métriques stables. Les distributions de classes, années, caméras et positions ont été maintenues au plus près ; l'écart maximal de proportion de classe observé dans le rapport de split n'est que de 0.0107 point.

Le split est group-aware : les vues Top et Bottom potentiellement associées restent dans le même jeu. Sans cette contrainte, une vue d'une châtaigne pourrait servir à l'apprentissage et l'autre à l'évaluation, ce qui rendrait les scores artificiellement optimistes. Les contrôles existants confirment des intersections train/validation/test vides et l'intégrité de tous les groupes.

> **Protection du test —** le test n'a servi ni au choix de l'architecture, ni aux hyperparamètres, au learning rate, au checkpoint ou au seuil. Il a été ouvert une seule fois après gel complet de la configuration.

## 4.2.3 Prétraitement et augmentation

Chaque image est lue en RGB, redimensionnée par interpolation bilinéaire en conservant son ratio, puis centrée dans un carré 224×224 par padding noir. Elle est ensuite convertie en tenseur et normalisée avec les statistiques ImageNet.

**Tableau 2 — Prétraitement commun et justification**

| Étape | Application | Justification CastagNet |
|---|---|---|
| RGB | Tous les jeux | Format d'entrée homogène à trois canaux. |
| 224×224 | Tous les jeux | Format standard torchvision, compromis information/coût. |
| Ratio conservé | Tous les jeux | Évite de déformer la forme du fruit. |
| Padding noir centré | Tous les jeux | Obtient un carré sans étirement artificiel. |
| Normalisation ImageNet | Tous les jeux | Cohérente avec les poids pré-entraînés. |
| Flip horizontal p=0,5 | Train uniquement | Variation réaliste sans changer la classe. |
| Rotation ±10° | Train uniquement | Simule une faible variation d'orientation. |
| Aucune augmentation | Validation et test | Mesure déterministe et comparable. |

*Source : training/transforms.py et configurations JSON*

Les augmentations restent volontairement légères. Des rotations fortes ou des transformations photométriques non justifiées auraient pu créer des textures ou des positions absentes du dispositif réel. La validation et le test utilisent strictement le même pipeline déterministe, sans flip ni rotation.

## 4.2.4 Baseline CNN maison

Un CNN maison sert de référence sans transfer learning. Son objectif n'est pas d'être optimal, mais de mesurer ce qu'une architecture légère entraînée depuis zéro peut apprendre sur CastagNet avant de mobiliser des représentations ImageNet.

L'architecture enchaîne quatre blocs Conv-BatchNorm-ReLU-MaxPool, avec 32, 64, 128 puis 256 canaux. Un AdaptiveAvgPool ramène les cartes à une valeur par canal, suivi d'un dropout de 0,3 et d'une couche linéaire 256→4. Aucun Softmax n'est placé dans forward, car CrossEntropyLoss attend directement les logits.

**Tableau 3 — Architecture du CNN maison**

| Composant | Configuration | Rôle |
|---|---|---|
| Bloc 1 | Conv 3→32, BN, ReLU, Pool | Premiers motifs visuels |
| Bloc 2 | Conv 32→64, BN, ReLU, Pool | Motifs intermédiaires |
| Bloc 3 | Conv 64→128, BN, ReLU, Pool | Caractéristiques plus abstraites |
| Bloc 4 | Conv 128→256, BN, ReLU, Pool | Représentation haute dimension |
| Sortie | AdaptiveAvgPool, Dropout 0,3, Linear 256→4 | Classification multiclasse |

*Source : training/models/custom_cnn.py*

Le réseau compte 389 924 paramètres, soit 1.4874 Mio en FP32. Cette petite taille en fait une baseline utile, mais limite aussi sa capacité à extraire les nuances visuelles entre classes proches.

CrossEntropyLoss non pondérée est adaptée aux quatre classes exclusives et garantit une comparaison identique entre expériences. AdamW a été choisi pour sa mise à jour robuste et sa séparation claire du weight decay. Le LR initial 0,001 constitue un point de départ standard pour cette baseline avec AdamW ; ReduceLROnPlateau et l'early stopping limitent les epochs inutiles.

## 4.2.5 Analyse des deux versions du CNN

**Tableau 4 — Comparaison des deux CNN maison sur validation**

| Modèle | LR | Best ep. | Accuracy | Macro-F1 | P Conf. | R Conf. | R NC | R PIETRA | Faux Conf. |
|---|---|---|---|---|---|---|---|---|---|
| CustomCNN v1 | 0,001 | 3 | 74,55 % | 62,68 % | 64,97 % | 94,15 % | 42,24 % | 16,80 % | 821 |
| CustomCNN v2 | 0,0003 | 3 | 77,88 % | 66,22 % | 64,74 % | 98,09 % | 64,71 % | 11,60 % | 835 |

*Source : metrics_summary.json des runs CustomCNN v1 et v2*

![Figure 1 — Losses d'entraînement et de validation des deux CNN maison](analysis/output/report_4_2_figures/customcnn_losses.png)

*Figure 1 — Losses d'entraînement et de validation des deux CNN maison. Source : training_history.csv des runs CustomCNN v1 et v2*

La v1 atteint un rappel Conforme élevé, mais au prix d'une précision faible et de 821 défauts NON Conforme ou PIETRA acceptés comme Conforme. Sa val_loss oscille fortement. L'audit du pipeline a validé model.eval(), l'absence de gradients en validation, shuffle=False, la remise à zéro des métriques, le calcul pondéré de la loss et l'intégrité du checkpoint : aucune fuite ou erreur d'évaluation n'a été trouvée.

Deux défauts de reporting ont néanmoins été corrigés pour les expériences suivantes : le LR était enregistré après scheduler.step, et le diagnostic d'overfitting comparait des epochs différentes. Ils n'ont modifié ni les poids ni les résultats v1.

La v2 ne change que le LR, abaissé à 0,0003 pour tester si une optimisation moins agressive stabilise l'apprentissage. Elle améliore la val_loss, l'accuracy, la macro-F1 et le rappel NON Conforme ; elle est donc retenue comme baseline maison. La v1 conserve légèrement moins de faux Conforme (821 contre 835), ce qui nuance le gain global de la v2.

> **Limite principale —** dans la v2, le rappel PIETRA n'est que de 11,60 % et 626 PIETRA sur 905 sont classées Conforme. Ce résultat indique d'abord une capacité de représentation insuffisante du petit CNN ; il ne démontre pas à lui seul un dataset défectueux.

## 4.2.6 Mise en place du transfer learning

Le dossier training annoncé dans le sujet était absent des données réellement fournies. Un pipeline simple et reproductible a donc été construit avec PyTorch, torchvision et MLflow, en conservant strictement les splits, la cible effective_label, le preprocessing, la seed 20260902 et les métriques.

Les poids ImageNet officiels torchvision fournissent des filtres visuels génériques déjà appris sur un vaste corpus. Pour un dataset d'environ 35 000 images, leur réutilisation permet d'éviter de réapprendre toutes les représentations depuis zéro et favorise une convergence plus efficace.

**Tableau 5 — Protocole commun de transfer learning**

| Phase | Paramètres entraînés | Epochs | LR | Régulation | Justification |
|---|---|---|---|---|---|
| 1 — tête | Tête seule ; backbone gelé | 3 | 0,001 | AdamW | Adapter rapidement la sortie aux 4 classes. |
| 2 — fine-tuning | Réseau entièrement dégelé | 15 max. | 0,0001 | Plateau + early stop 5 | Ajuster progressivement les poids ImageNet. |

*Source : training/train.py et configurations des modèles pré-entraînés*

Le LR plus faible en fine-tuning évite de détruire brutalement les représentations pré-entraînées. CrossEntropyLoss reste non pondérée afin d'isoler l'effet de l'architecture et de conserver un protocole comparable. Aucun grand balayage d'hyperparamètres n'a été entrepris, ce qui limite le risque de sur-optimisation sur validation.

## 4.2.7 Modèles pré-entraînés étudiés

**Tableau 6 — Complexité théorique et statut des architectures**

| Modèle | Paramètres | Taille FP32 | Entraîné | Positionnement |
|---|---|---|---|---|
| CustomCNN v2 | 389 924 | 1,4874 Mio | Oui | Baseline légère |
| MobileNetV3 Small | 1 521 956 | 5,8058 Mio | Oui | Représentant edge |
| EfficientNet-B0 | 4 012 672 | 15,3071 Mio | Oui | Compromis intermédiaire |
| ResNet18 | 11 178 564 | 42,6428 Mio | Oui | Candidat qualité |
| ShuffleNetV2 x1.0 | 1 257 704 | ≈ 4,80 Mio | Non | Préparé uniquement |

*Source : smoke_test_summary.json, factory torchvision et configurations*

MobileNetV3 Small représente l'option légère pertinente pour une future cible edge. EfficientNet-B0 fournit un point intermédiaire entre taille et qualité. ResNet18, plus lourd, sert de candidat orienté qualité. Ces rôles permettent une comparaison utile sans supposer qu'un modèle plus petit est automatiquement plus rapide sur le matériel réel.

ShuffleNetV2 x1.0 a été préparé mais volontairement non entraîné. MobileNet couvrait déjà le besoin d'un représentant très léger ; la comparaison comprenait alors une baseline, un modèle edge, un intermédiaire et un candidat qualité. Une expérience supplémentaire aurait apporté une valeur limitée au regard du temps et accru le risque de recherche excessive sur validation.

## 4.2.8 Suivi des expériences avec MLflow

MLflow centralise configurations, hyperparamètres, learning rates, losses, accuracy, macro-F1, checkpoints, durées et artefacts. Il permet de comparer les modèles avec une trace reproductible et relie explicitement le checkpoint final au run qui l'a produit.

**Tableau 7 — Traçabilité des runs d'entraînement MLflow**

| Modèle | Run ID | Epochs | Best ep. | Durée | Statut |
|---|---|---|---|---|---|
| CustomCNN v1 | 76cb13166116439aa05a4174fd954c18 | 8 | 3 | 1 h 00 min 31 s | FINISHED |
| CustomCNN v2 | 42420caec760477da2c2cdb4c1041c10 | 8 | 3 | 1 h 05 min 23 s | FINISHED |
| MobileNetV3 Small | 8b9b161b9dd24e208af9be3baa532f2e | 16 | 11 | 2 h 26 min 17 s | FINISHED |
| EfficientNet-B0 | a434cccec8624ad2bcdae5fe952726e9 | 15 | 10 | 2 h 37 min 44 s | FINISHED |
| ResNet18 | db0ee7873903411e899e2f85384ae58a | 18 | 13 | 2 h 31 min 27 s | FINISHED |

*Source : mlflow.db et metrics_summary.json*

Pour ResNet18, l'historique distingue explicitement les trois epochs tête seule des quinze epochs de fine-tuning. Le checkpoint de l'epoch 13 minimise la val_loss à 0.325563. Le run final est db0ee7873903411e899e2f85384ae58a.

![Figure 2 — ResNet18 — losses train et validation, phases et meilleur checkpoint](analysis/output/report_4_2_figures/resnet18_losses.png)

*Figure 2 — ResNet18 — losses train et validation, phases et meilleur checkpoint. Source : analysis/output/modeling/resnet18/training_history.csv*

![Figure 3 — ResNet18 — accuracy train et validation](analysis/output/report_4_2_figures/resnet18_accuracy.png)

*Figure 3 — ResNet18 — accuracy train et validation. Source : analysis/output/modeling/resnet18/training_history.csv*

![Figure 4 — ResNet18 — macro-F1 de validation](analysis/output/report_4_2_figures/resnet18_val_macro_f1.png)

*Figure 4 — ResNet18 — macro-F1 de validation. Source : analysis/output/modeling/resnet18/training_history.csv*

Les courbes montrent le gain du fine-tuning après adaptation de la tête. La hausse finale de la performance train alors que la validation plafonne justifie la sélection du meilleur checkpoint plutôt que de la dernière epoch. L'écart d'accuracy au meilleur checkpoint reste limité à 4,57 points selon l'artefact.

## 4.2.9 Comparaison des architectures

**Tableau 8 — Comparaison principale sur validation**

| Modèle | Param. | Acc. | Macro-F1 | P Conf. | R Conf. | R NC | R PIETRA | Faux Conf. | Conf. rejetés |
|---|---|---|---|---|---|---|---|---|---|
| CustomCNN v2 | 389 924 | 77,88 % | 66,22 % | 64,74 % | 98,09 % | 64,71 % | 11,60 % | 835 | 31 |
| MobileNetV3 Small | 1 521 956 | 84,21 % | 80,49 % | 87,31 % | 80,43 % | 76,24 % | 67,51 % | 187 | 318 |
| EfficientNet-B0 | 4 012 672 | 86,61 % | 83,60 % | 87,78 % | 83,57 % | 76,82 % | 75,14 % | 184 | 267 |
| ResNet18 | 11 178 564 | 88,15 % | 85,21 % | 88,36 % | 87,82 % | 74,59 % | 78,45 % | 181 | 198 |

*Source : metrics_summary.json de chaque modèle*

Le transfer learning produit un saut qualitatif net. Par rapport au CustomCNN v2, MobileNet améliore déjà fortement la macro-F1 et la reconnaissance de PIETRA. EfficientNet progresse encore, puis ResNet18 obtient la meilleure accuracy, la meilleure macro-F1, la meilleure précision et le meilleur rappel Conforme, le meilleur rappel PIETRA et le plus faible compteur métier de faux Conforme.

MobileNetV3 Small reste attractif par sa taille, mais son rappel Conforme de 80,43 % est inférieur au minimum de 85 %. EfficientNet-B0 constitue un compromis plus performant, avec un rappel Conforme de 83,57 %, mais demeure légèrement derrière ResNet18 sur les critères prioritaires.

## 4.2.10 Sélection de ResNet18

ResNet18 est retenu comme meilleur candidat prédictif du §4.2. Il est le seul modèle pré-entraîné évalué dont le rappel Conforme dépasse 85 % avec la décision argmax, tout en obtenant les meilleurs indicateurs globaux et le plus faible nombre de défauts NON Conforme ou PIETRA acceptés comme Conforme.

Le modèle compte 11 178 564 paramètres, contre 1,52 million pour MobileNet et 4,01 millions pour EfficientNet. Cette différence n'est pas masquée : ResNet18 est choisi comme candidat qualité parce que la protection de la classe Conforme est prioritaire. Son coût embarqué devra néanmoins être démontré, et non supposé.

> **Portée de la sélection —** ResNet18 est retenu comme candidat qualité à l'issue du §4.2. Son adéquation définitive à la cible matérielle reste conditionnée aux benchmarks d'inférence du §4.3.

## 4.2.11 Calibration du seuil Conforme

La calibration a été conduite uniquement sur les 5 289 images de validation, sans réentraînement. Pour chaque seuil de 0,50 à 0,99 par pas de 0,01, une image est classée Conforme si P(Conforme) atteint le seuil ; sinon, l'argmax est calculé seulement parmi NON Conforme, PIETRA et Vide.

**Tableau 9 — Trois seuils de décision examinés sur validation**

| Seuil | P Conforme | R Conforme | Macro-F1 | Faux Conf. | Conf. rejetés | Décision |
|---|---|---|---|---|---|---|
| 0,55 | 89,99 % | 85,23 % | 85,07 % | 150 | 240 | Retenu |
| 0,56 | 90,02 % | 84,92 % | 85,01 % | 149 | 245 | Rappel < 85 % |
| 0,91 | 95,31 % | 63,75 % | 80,37 % | 49 | 589 | Rappel insuffisant |

*Source : conforme_threshold_curve.csv — validation uniquement*

![Figure 5 — Calibration du seuil de décision Conforme](analysis/output/report_4_2_figures/resnet18_conforme_threshold_calibration.png)

*Figure 5 — Calibration du seuil de décision Conforme. Source : analysis/output/modeling/resnet18/calibration/conforme_threshold_curve.csv*

Aucun des 50 seuils ne satisfait simultanément 95 % de précision et 85 % de rappel. Le compromis mathématique minimal se situe à 0,56, avec 90,02 % de précision et 84,92 % de rappel. Comme le rappel passe sous le minimum métier, ce seuil est rejeté.

Le seuil 0,55 conserve 85,23 % de rappel pour 89,99 % de précision. Le gain de précision apporté par 0,56 est négligeable face au franchissement de la contrainte de rappel ; 0,55 a donc été figé avant l'ouverture du test.

Le premier seuil atteignant 95 % de précision est 0,91, mais le rappel chute à 63,75 %. Ce compromis est incompatible avec l'exigence de rendement minimal. Plus généralement, le chevauchement des scores entre Conforme et les autres classes empêche un simple seuil d'atteindre les deux objectifs.

## 4.2.12 Évaluation finale sur le jeu de test

Le test a été ouvert une seule fois après gel du modèle ResNet18, du checkpoint de l'epoch 13, du run MLflow et du seuil 0,55. Les 5 288 images ont fait l'objet d'une passe d'inférence unique ; les décisions argmax et seuil réutilisent exactement les mêmes probabilités. Aucun entraînement, recalibrage ou changement n'a suivi.

**Tableau 10 — ResNet18 argmax — validation et test**

| Jeu | Accuracy | Macro-F1 | P Conforme | R Conforme | Faux Conf. | Conf. rejetés |
|---|---|---|---|---|---|---|
| Validation | 88,15 % | 85,21 % | 88,36 % | 87,82 % | 181 | 198 |
| Test | 87,71 % | 84,54 % | 88,15 % | 87,45 % | 188 | 204 |

*Source : ResNet18 metrics_summary.json et final_test_metrics.json*

La proximité des deux lignes indique une généralisation cohérente sur ce dataset : l'accuracy diminue d'environ 0,44 point et la macro-F1 de 0,68 point. Cette stabilité ne doit toutefois pas être extrapolée à de nouvelles campagnes, d'autres lots ou au fonctionnement en production.

**Tableau 11 — Test final — argmax et règle métier figée**

| Décision | Accuracy | Macro-P | Macro-R | Macro-F1 | P Conf. | R Conf. | Faux Conf. | Conf. rejetés |
|---|---|---|---|---|---|---|---|---|
| Argmax | 87,71 % | 84,99 % | 84,36 % | 84,54 % | 88,15 % | 87,45 % | 188 | 204 |
| Seuil 0,55 | 87,50 % | 84,80 % | 84,44 % | 84,44 % | 89,16 % | 85,54 % | 166 | 235 |

*Source : analysis/output/modeling/resnet18/final_test/final_test_metrics.json*

**Tableau 12 — Métriques par classe — configuration finale test**

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| Conforme | 89,16 % | 85,54 % | 87,31 % | 1 625 |
| NON Conforme | 85,24 % | 76,12 % | 80,42 % | 850 |
| PIETRA | 65,69 % | 76,80 % | 70,81 % | 905 |
| Vide | 99,11 % | 99,32 % | 99,21 % | 1 908 |

*Source : final_test_metrics.json — threshold_055*

![Figure 6 — Matrice de confusion ResNet18 sur test — argmax](analysis/output/report_4_2_figures/test_confusion_argmax.png)

*Figure 6 — Matrice de confusion ResNet18 sur test — argmax. Source : final_test_metrics.json — argmax*

![Figure 7 — Matrice de confusion ResNet18 sur test — seuil Conforme 0,55](analysis/output/report_4_2_figures/test_confusion_threshold_055.png)

*Figure 7 — Matrice de confusion ResNet18 sur test — seuil Conforme 0,55. Source : final_test_metrics.json — threshold_055*

Avec le seuil 0,55, 36 NON Conforme et 130 PIETRA sont acceptées comme Conforme, contre respectivement 42 et 146 en argmax. Le compteur métier baisse donc de 188 à 166, au prix de 31 rejets Conforme supplémentaires. Ce déplacement est cohérent avec la priorité qualité : l'acceptation devient plus stricte sans dégrader fortement les métriques globales.

La confusion dominante reste Conforme↔PIETRA : 200 vrais Conforme sont classés PIETRA et 130 PIETRA deviennent Conforme avec la règle finale. La classe Vide reste très bien séparée. Le compteur « faux Conforme métier » additionne volontairement NON Conforme et PIETRA ; les trois Vide prédits Conforme restent inclus dans la précision statistique.

## 4.2.13 Analyse du respect du cahier des charges

**Tableau 13 — Respect des contraintes métier sur le test final**

| Critère | Objectif | Résultat | Écart | Verdict |
|---|---|---|---|---|
| Précision Conforme | ≥ 95 % | 89,16 % | −5,84 pts | NON ATTEINT |
| Rappel Conforme | ≥ 85 % | 85,54 % | +0,54 pts | ATTEINT |
| Respect simultané | Deux critères | Un critère sur deux | — | NON ATTEINT |

*Source : final_test_metrics.json — business_requirements*

> **Conclusion obligatoire —** Le modèle final respecte l'objectif minimal de rappel de la classe Conforme mais ne satisfait pas l'objectif de précision de 95 %.

Ce résultat est présenté sans ajustement post-test. Rechercher maintenant un autre seuil, checkpoint ou modèle à partir de ces 5 288 images transformerait le test en jeu de validation et invaliderait l'estimation finale. Le test est donc fermé à toute optimisation future.

## 4.2.14 Prise en compte des contraintes d'embarquement

La complexité théorique place ResNet18 au-dessus de MobileNetV3 Small et EfficientNet-B0. Ce constat fournit une indication de stockage et de mémoire des poids, mais ne permet pas de conclure sur la latence réelle : les opérateurs, le runtime, la précision numérique et le matériel influencent directement le débit.

MobileNet est plus petit, mais n'est pas automatiquement plus rapide dans toutes les conditions. Inversement, les 42,64 Mio FP32 de ResNet18 ne prouvent pas qu'il est incompatible avec 3 Go de VRAM. Le §4.3 devra exporter en ONNX et mesurer taille, latence batch 1, débit, mémoire CPU/GPU et capacité à traiter douze flux sur ou au plus près de la GTX 1060 3 Go et de l'ancien i7.

## 4.2.15 Limites, recul critique et pistes d'amélioration

### Limites observées

- La précision Conforme finale de 89,16 % reste inférieure aux 95 % demandés.
- PIETRA demeure la classe la plus difficile et partage une frontière visuelle avec Conforme.
- Les labels proviennent d'un processus humain historiquement hétérogène, malgré l'adjudication du §4.1.
- ResNet18 est le plus lourd des trois modèles pré-entraînés réellement évalués.
- La performance ONNX, la GTX 1060 3 Go et les douze flux n'ont pas encore été testés.
- Un seul dataset est disponible ; la robustesse sur une collecte indépendante reste inconnue.
- Le dossier training officiel annoncé dans le sujet était absent des données fournies.
- Le nombre d'architectures et le tuning ont été volontairement limités.
- Le test final est désormais fermé à toute optimisation.

Les ambiguïtés Conforme/PIETRA/NON Conforme peuvent provenir à la fois de la difficulté visuelle, de la variabilité des acquisitions et des conventions de labellisation. L'amélioration spectaculaire entre CNN maison et transfer learning montre toutefois que la capacité du modèle expliquait une part importante des erreurs initiales.

### Pistes futures — hors résultats du §4.2

- Harmoniser la labellisation et revoir de façon ciblée les PIETRA difficiles et les faux Conforme.
- Étudier une CrossEntropy pondérée, un coût métier asymétrique ou un WeightedRandomSampler.
- Évaluer une calibration probabiliste plus avancée sur de nouvelles données de validation.
- Collecter un jeu indépendant et étudier une approche hiérarchique Vide/Fruit puis qualité.
- Tester une autre résolution d'entrée uniquement dans un nouveau protocole expérimental.
- Évaluer quantification, compression et éventuellement distillation après le benchmark ONNX.
- Mesurer sur le matériel cible et sur douze flux avant tout choix embarqué définitif.

Les expériences s'arrêtent ici de manière raisonnée. Deux CNN maison, trois modèles pré-entraînés, trois niveaux de complexité, une calibration validation et un test final unique suffisent pour répondre à l'objectif méthodologique. Multiplier les essais apporterait une valeur décroissante et favoriserait la sur-optimisation sur validation.

## 4.2.16 Conclusion et transition vers le §4.3

Le §4.2 a établi un pipeline reproductible malgré l'absence du dossier training annoncé, construit une baseline CNN, démontré l'intérêt du transfer learning et comparé des architectures de complexités différentes sous un protocole commun. MLflow relie chaque résultat à sa configuration et au checkpoint correspondant.

ResNet18 est le meilleur candidat prédictif sur validation. Le seuil 0,55, choisi avant le test, réduit les défauts acceptés comme Conforme tout en maintenant le rappel au-dessus de 85 %. L'évaluation finale reste cohérente avec la validation, mais la précision de 95 % n'est pas atteinte : le cahier des charges est donc seulement partiellement satisfait.

La prochaine étape ne consiste pas à ajuster le modèle au test. Le §4.3 devra vérifier si le candidat qualité peut être exporté en ONNX et respecter les contraintes de latence, débit et mémoire sur la cible envisagée. Cette mesure décidera si ResNet18 peut être retenu tel quel ou si une optimisation embarquée, voire un modèle plus léger, doit être étudié sur un nouveau protocole.

## Annexe — Registre des artefacts utilisés

Les valeurs du rapport proviennent exclusivement des artefacts suivants. Les nombres arrondis dans le texte sont calculés à partir de leurs valeurs exactes.

- analysis/output/final_training_manifest_summary.json
- analysis/output/ml_split_summary.json
- analysis/output/modeling/*/config.json
- analysis/output/modeling/*/smoke_test_summary.json
- analysis/output/modeling/*/metrics_summary.json
- analysis/output/modeling/*/training_history.csv
- analysis/output/modeling/resnet18/calibration/conforme_threshold_curve.csv
- analysis/output/modeling/resnet18/calibration/conforme_threshold_summary.json
- analysis/output/modeling/resnet18/final_test/final_test_metrics.json
- analysis/output/modeling/resnet18/final_test/confusion_matrix_argmax.csv
- analysis/output/modeling/resnet18/final_test/confusion_matrix_threshold_055.csv
- mlflow.db — expérience CastagNet_4_2 et runs cités.

### Points de cohérence et conventions

- Les artefacts exacts priment sur les valeurs arrondies du cahier de consignes.
- Le compromis automatisé de calibration était 0,56 ; le choix métier final documenté est 0,55 afin de préserver le rappel minimal.
- Le diagnostic overfitting_probable de CustomCNN v1 provient d'un ancien défaut de reporting ; les poids et métriques ne sont pas affectés.
- Le compteur faux Conforme total des artefacts additionne NON Conforme et PIETRA ; les Vide→Conforme sont néanmoins comptés dans la précision.
- ShuffleNetV2 x1.0 est configuré mais ne possède aucun résultat d'entraînement.
