# Calibration du seuil Conforme — ResNet18

Calibration effectuée uniquement sur les 5289 images de validation avec le
checkpoint inchangé de l'epoch 13. Le manifeste test n'a pas été chargé.

## Règle

Si `P(Conforme) >= seuil`, la prédiction est Conforme. Sinon, elle est l'argmax
parmi NON Conforme, PIETRA et Vide. Les 50 seuils de 0,50 à 0,99 ont été testés.

## Résultat

- Seuils satisfaisant simultanément les deux objectifs : 0
- Méthode de sélection : minimum_euclidean_shortfall

- `threshold` : 0.56
- `precision_conforme` : 0.9001956947162426
- `recall_conforme` : 0.8492307692307692
- `f1_conforme` : 0.8739708676377455
- `non_conforme_to_conforme` : 38
- `pietra_to_conforme` : 111
- `false_conforme_total` : 149
- `conforme_rejected` : 245
- `accuracy` : 0.8780487804878049
- `macro_f1` : 0.8500795568508785
- `precision_shortfall` : 0.049804305283757344
- `recall_shortfall` : 0.0007692307692307443
- `distance_to_objectives` : 0.049810245339428225

En l'absence de seuil admissible, la distance est la norme euclidienne des seuls
écarts manquants à 95 % de précision et 85 % de rappel. Aucun réentraînement et
aucune modification du checkpoint, des images ou des splits n'ont été effectués.
