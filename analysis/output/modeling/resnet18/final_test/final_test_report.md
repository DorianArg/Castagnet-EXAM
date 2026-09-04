# Évaluation finale unique — test CastagNet

Modèle ResNet18, checkpoint epoch 13 du run MLflow `db0ee7873903411e899e2f85384ae58a`.
Le test compte 5288 images et a fait l'objet d'une seule passe d'inférence.

Le test n'a servi à aucune sélection d'architecture ni aucun réglage
d'hyperparamètre. Le seuil Conforme `0.55` a été déterminé
uniquement sur validation avant l'ouverture du test. Il n'est pas adapté selon
les résultats ci-dessous. Cette exécution constitue l'évaluation finale unique.

## Classification standard — argmax

- Loss : 0.346875
- Accuracy : 87.7080%
- Macro precision : 84.9929%
- Macro recall : 84.3581%
- Macro F1 : 84.5364%
- Precision Conforme : 88.1514%
- Recall Conforme : 87.4462%
- F1 Conforme : 87.7973%
- NON Conforme → Conforme : 42
- PIETRA → Conforme : 146
- Faux Conforme total : 188
- Conforme rejetées : 204


## Configuration métier — seuil Conforme 0,55

- Loss : 0.346875
- Accuracy : 87.5000%
- Macro precision : 84.8011%
- Macro recall : 84.4426%
- Macro F1 : 84.4397%
- Precision Conforme : 89.1597%
- Recall Conforme : 85.5385%
- F1 Conforme : 87.3116%
- NON Conforme → Conforme : 36
- PIETRA → Conforme : 130
- Faux Conforme total : 166
- Conforme rejetées : 235


## Contraintes métier sur la configuration figée

- Precision Conforme ≥ 95 % : FAIL
- Recall Conforme ≥ 85 % : PASS
- Respect simultané : FAIL

Quelle que soit cette conclusion, aucun ajustement du modèle ou du seuil ne doit
être réalisé à partir du jeu test.
