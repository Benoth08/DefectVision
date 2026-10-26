"""DefectVision : controle qualite visuel de pieces moulees (OK / defaut).

Trois modeles :
- supervise : ResNet-18 en transfert d'apprentissage ;
- non supervise : autoencodeur entraine uniquement sur des pieces saines ;
- hybride : combinaison des deux scores.
"""

__version__ = "1.0.0"
