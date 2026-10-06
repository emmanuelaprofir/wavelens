Extraction de fréquences radio, visualisation, analyse, et extraction de données.

Utilisation du repository kiwiclient -> https://github.com/jks-prv/kiwiclient
Avant tout il faut installer les dépendances renseignées dans leur README.md

Les fichiers que contient kiwi_data ont été généré par kiwi_analyse.py, c'est un exemple de ce que l'on peut obtenir
avec la fréquence que j'ai récupéré.

J'ai choisi un WebSDR (Software-Defined Radio receiver connected to the internet) sur le site http://kiwisdr.com/public/ .
Il faut veiller à ce que le SDR ne requiert pas d'authentification avec un mot de passe, et qu'il reste une place.

Il faut cloner le repository kiwiclient et déplacer kiwi_analyse.py dedans. 
Ensuite exécuter dans kiwiclient -> "python3 kiwi_analyse.py"

Explication par étapes :

1. Connexion au SDR (kiwiclient/kiwirecorder.py et kiwiclient/kiwi/client.py)

2. Authentification avec un mot de passe vide

3. Ecoute de la radio pendant 30 sec autour de la fréquence centrale choisie 

4. Exportation en .wav avec les échantillons IQ (In phase et Quadrature) (kiwiclient/kiwirecorder.py)

5. Lecture et décodage du fichier IQ (-> utilisation de kiwi/wavreader.py)

6. Extraction des métadonnées du signal

7. Analyse FFT du signal (convertit les signaux radio du domaine temporel vers le domaine fréquentiel)

8. Détection des pics fréquentiels

9. Estimation du niveau de bruit et du SNR (pour estimer la détectabilité du signal)

10. Calcul de la bande passante (plage de fréquences occupée)

11. Analyse de l'évolution du signal dans le temps (blocks.csv)

12. Génération du spectre et du spectrogramme

13. Sauvegarde des données IQ et des résultats d'analyse.

NB: Dans un .wav de 30secondes, il y a environ 360 000 échantillons avec I et Q.