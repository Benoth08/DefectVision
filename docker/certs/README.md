# Certificats racine supplémentaires (facultatif)

Si un proxy d'entreprise ou un antivirus inspecte le HTTPS du poste, les
téléchargements faits pendant la construction des images (`pip`) et
l'acquisition des données échouent en vérification TLS.

Déposer ici la partie publique du certificat racine concerné, au format PEM,
avec l'extension `.crt`. Les Dockerfiles l'ajoutent aux autorités reconnues
(`update-ca-certificates`). Les fichiers `.crt` de ce dossier sont ignorés par
git : ils ne sont jamais versionnés.

Ne jamais désactiver la vérification TLS à la place.
