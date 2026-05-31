[![Demonstration de l'application](static/uploads/login.png)](https://drive.google.com/file/d/1rvUykfPpx25QyNDN_ROdvFwrA7vSMgzv/view?usp=sharing)
*Cliquez sur l'image ci-dessus pour lancer la video de demonstration complete.*

Application Clinique de Detection du Cancer de la Peau par Vision Transformer (ViT)

Cette application web medicale avancee combine la puissance des Vision Transformers (ViT) et un backend robuste sous Flask pour offrir un outil d'aide a la decision clinique. Elle permet aux dermatologues et professionnels de sante d'analyser des images de lesions cutanees suspectes afin d'evaluer rapidement la probabilite qu'elles soient benignes ou malignes.

1. Introduction et Role de l'Application

Le melanome et les autres cancers de la peau representent une part majeure des diagnostics oncologiques mondiaux. Detectes a un stade precoce, le taux de survie depasse 95%. Cependant, le diagnostic visuel humain requiert des annees d'expertise et reste sujet a l'erreur d'interpretation.

Cette application a ete concue pour combler ce fosse en mettant a disposition un systeme d'inference dermatologique automatise.

Role Cle de l'Outil :

Assistance au Triage : Permettre un pre-depistage rapide pour orienter les cas prioritaires vers des examens histopathologiques (biopsies).

Fiabilite Mathematique : S'appuyer sur l'architecture Vision Transformer (ViT), qui excelle dans l'analyse globale et locale des images en modelisant les relations spatiales a longue distance entre les pixels d'une lesion (bordures, asymetrie, variations de couleur).

Suivi Patient Integre : Fournir un carnet de bord numerique securise pour suivre l'evolution des diagnostics d'une cohorte de patients.

2. Visite Guidee de l'Interface (Description des Pages)

L'application propose un parcours utilisateur fluide, ergonomique et securise, decoupe en plusieurs ecrans specialises :

Pages d'Acces : Connexion (/login) & Inscription (/register)

Ce que l'on y voit : Des formulaires epures demandant un identifiant de praticien et un mot de passe (avec double saisie securisee pour l'inscription).

Utilite : Garantir que seuls les medecins et personnels cliniques autorises accedent aux donnees sensibles des patients et a la console de diagnostic. Les sessions utilisateur sont cryptees grace a une cle secrete.

Page de Recuperation : Mot de passe oublie (/forgot_password)

Ce que l'on y voit : Un champ de saisie pour l'adresse e-mail ou l'identifiant professionnel.

Utilite : Simuler un protocole de securite en cas de perte des identifiants d'acces. Elle affiche un message confirmant l'envoi des instructions de recuperation, assurant une experience utilisateur complete et professionnelle lors des presentations et demonstrations.

Page d'Accueil : Le Tableau de Bord (/dashboard)

Ce que l'on y voit : Une interface d'accueil affichant un message de bienvenue personnalise, des statistiques rapides et deux grandes cartes interactives : "Nouvelle Analyse" et "Historique des Patients".

Utilite : Il s'agit du hub central de navigation. Il permet au medecin de choisir instantanement son action des qu'il se connecte.

Page de Diagnostic : Nouvelle Analyse (/predict)

Ce que l'on y voit : Un formulaire pour saisir le nom et l'age du patient, couple a une zone interactive de televersement (Drag & Drop) pour charger la photo macroscopique ou dermatoscopique de la lesion cutanee.

Utilite : Collecter l'ensemble des donnees d'un examen medical clinique avant d'appeler l'intelligence artificielle pour l'evaluation.

Page des Resultats : Rapport de Diagnostic (/result/)

Ce que l'on y voit : Un rapport medical structure presentant la photo analysee a cote du verdict final ("Benigne" ou "Maligne") accompagne d'une jauge visuelle coloree affichant le pourcentage exact de certitude de l'IA.

Utilite : Fournir une reponse immediate et facilement interpretable pour le praticien afin d'alimenter son dossier medical.

Page d'Archivage : Historique Clinique (/patients_history)

Ce que l'on y voit : Un grand tableau chronologique repertoriant tous les patients analyses, leurs photos respectives, leurs ages, les dates precises des examens, ainsi que les verdicts et indices de confiance de l'IA.

Utilite : Assurer la tracabilite des diagnostics, permettre des retours sur dossiers et faciliter le suivi a long terme de l'evolution des lesions chez les patients.

3. Entrainement du Modele sur Google Colab (Deep Learning)

Pour creer notre intelligence artificielle, nous avons utilise Google Colab. Cela nous a permis de profiter gratuitement de leurs cartes graphiques (GPU) pour que l'entrainement soit beaucoup plus rapide avant d'integrer le resultat dans notre site web Flask.

Voici comment nous avons procede de notre cote, etape par etape :

Recuperation des donnees : Nous avons d'abord rassemble un grand nombre d'images de lesions cutanees en utilisant des bases de donnees medicales fiables (comme HAM10000 ou ISIC). Chaque image que nous avons utilisee etait deja validee par des medecins comme etant "benigne" ou "maligne".

Preparation des images (Data Augmentation) : Pour eviter que notre IA n'apprenne les photos par coeur (ce qu'on appelle l'overfitting), nous avons utilise Keras pour modifier legerement les images d'entrainement. Concretement, nous avons applique des rotations aleatoires, des zooms et des petits changements de contraste pour habituer l'IA a analyser des photos de differentes qualites. Ensuite, nous avons redimensionne toutes les images en 224x224 pixels pour qu'elles aient la meme taille.

Utilisation d'un modele existant (Transfer Learning) : Plutot que de coder un reseau de neurones en partant de zero, nous avons pris la decision d'importer un modele tres puissant appele Vision Transformer (ViT) depuis TensorFlow Hub. Nous avons "gele" la majeure partie de ce modele pour conserver sa capacite de base a reconnaitre des formes et des textures. Ensuite, nous lui avons simplement ajoute notre propre derniere couche (avec une fonction d'activation Sigmoide) pour qu'il se concentre uniquement sur notre besoin : classer les maladies de la peau.

L'entrainement : Nous avons lance l'entrainement sur nos images pendant plusieurs cycles (epoques). Nous avons utilise l'optimisateur Adam et nous surveillions sa precision globale a chaque etape pour nous assurer qu'il apprenait correctement a differencier les deux types de lesions.

Sauvegarde du resultat : Une fois que le modele etait bien entraine et que sa precision etait satisfaisante, nous avons exporte son apprentissage sous forme d'un simple fichier (modele_poids.weights.h5). C'est ce fichier que nous avons place dans notre dossier de projet, et qui permet aujourd'hui a notre application web de faire des predictions en direct sans avoir a recalculer quoi que ce soit.


Etape 1 : Preparation du Dataset

Nous avons utilise des datasets medicaux de reference (comme HAM10000 ou la base ISIC), contenant des milliers d'images dermatoscopiques labellisees de melanomes (malins) et de naevus/keratoses (benins).

Etape 2 : Data Augmentation (Augmentation de donnees)

Afin d'eviter le surapprentissage (overfitting) et de rendre le modele robuste a n'importe quelle luminosite ou angle de prise de vue, nous avons applique des transformations dynamiques aux images d'entrainement via Keras :

Rotations aleatoires

Zooms et cisaillements legers

Flous gaussiens et ajustements de contraste

Etape 3 : Transfer Learning (Apprentissage par transfert)

Nous avons importe l'architecture Vision Transformer (ViT-B/16) pre-entrainee sur ImageNet via TensorFlow Hub.

Les couches profondes d'extraction de caracteristiques geometriques et de motifs ont ete gelees (non entrainables) pour conserver les connaissances fondamentales de l'IA.

Nous avons ajoute une couche dense terminale avec une fonction d'activation Sigmoide dediee a notre probleme de classification binaire (Benin vs Malin).

Etape 4 : Compilation et Entrainement

Le modele a ete compile avec l'optimisateur Adam (avec un taux d'apprentissage initial tres faible de 0.0001 pour ne pas alterer les poids pre-entraines) et la fonction de perte Binary Crossentropy (entropie croisee binaire).
L'entrainement a ete mene sur plusieurs epoques en mesurant la precision (Accuracy) et l'aire sous la courbe (AUC) de validation.

Etape 5 : Exportation des poids

Une fois la convergence atteinte, nous avons extrait uniquement la matrice de connaissances apprise par le reseau de neurones et l'avons sauvegardee dans un fichier de poids leger : modele_poids.weights.h5. C'est ce fichier qui est directement charge par notre application Flask en local.

4. Guide de Lancement Rapide (Local)

Etape 1 : Prerequis systeme

Installer Python 3.12 (ou superieur).

Installer XAMPP ou WAMP sur votre machine.

Etape 2 : Installation des paquets requis

Dans votre invite de commande (dans le dossier SKIN_CANCER_APP), lancez :

pip install flask flask_sqlalchemy pymysql cryptography tensorflow tensorflow_hub numpy werkzeug


Etape 3 : Initialisation de la base de donnees

Ouvrez votre panneau de controle XAMPP et demarrez les modules Apache et MySQL.

Allez sur votre navigateur a l'adresse : http://localhost/phpmyadmin/

Creez une base de donnees vide nommee precisement : skin_cancer_db (interclassement utf8mb4_general_ci).

Note technique : Ne creez pas de tables manuellement. Le bloc db.create_all() present dans app.py genere automatiquement les tables requises lors du premier demarrage du serveur.

Etape 4 : Depot du modele d'IA

Creez un dossier nomme model a la racine de votre application.

Placez-y votre fichier de poids entraine sur Google Colab nomme : modele_poids.weights.h5.

Etape 5 : Lancement

Executez l'application en ecrivant :

python app.py


Apres environ 15 a 30 secondes (temps necessaire au chargement en memoire vive du Vision Transformer), ouvrez votre navigateur sur : http://127.0.0.1:5000

5. Anatomie et Explication de app.py

Le fichier app.py est le coeur logique de notre application. Voici le detail de ses composants cles :

Partie 1 : Declaration et Configurations de l'Application

app.secret_key : Une cle cryptographique obligatoire permettant de securiser la session du navigateur, empechant ainsi l'usurpation d'identite de session.

SQLALCHEMY_DATABASE_URI : Specifie l'adresse d'acces a MySQL. pymysql sert de traducteur entre notre code Python et la base locale geree par XAMPP.

Partie 2 : Integration du Vision Transformer (ViT)

La Classe HubFeatureExtractor : Un calque (layer) personnalise Keras indispensable pour envelopper proprement le bloc d'extraction de caracteristiques issu de TensorFlow Hub, assurant sa compatibilite lors des phases de prediction.

Chargement de l'architecture : Nous reconstruisons a la volee le reseau en associant l'extracteur ViT de Google avec une couche finale Dense comprenant un seul neurone de sortie configure en sigmoide (pour sortir un score continu compris entre 0 et 1).

load_weights : Si le fichier est trouve dans le dossier model, l'IA integre les connaissances geometriques apprises durant l'entrainement cloud sur Google Colab.

Partie 3 : Modelisation de Donnees (ORM)

Au lieu d'ecrire du code SQL textuel, Flask utilise des classes Python pour schematiser la base de donnees.

User stocke de facon securisee les identifiants cliniques.

Patient consigne l'ensemble des donnees d'un diagnostic medical, y compris le chemin d'acces local vers l'image televersee, l'age du patient et le pourcentage de certitude calcule par la couche d'activation sigmoide.

Partie 4 : Algorithmes Metiers et Routage Dynamique

Authentification (/login & /register) : Gere la connexion des medecins. Une fois authentifie, le systeme attribue au medecin une variable globale en session qui sert de pass d'entree.

Verification de securite (Sessions) : Chaque page sensible commence par verifier si la session contient bien cet identifiant. Si ce n'est pas le cas, l'utilisateur est redirige vers la page de connexion.

Le Pretraitement d'Inference (/submit_analysis) :

Il receptionne l'image et securise son nom d'ecriture.

Il l'enregistre sur le serveur avec un prefixe horodate unique pour eviter tout ecrasement accidentel.

L'image est redimensionnee en 224x224 pixels et normalisee en divisant ses canaux de couleur par 255.0 pour obtenir des valeurs entre 0 et 1.

L'IA calcule une probabilite d'analyse. Si cette valeur est superieure a 50% (0.5), la lesion est etiquetee comme maligne (Malignant). Sinon, elle est classee comme benigne (Benign). Le taux de confiance affiche a l'ecran correspond au pourcentage de certitude calcule pour le diagnostic retenu.

Toutes ces informations sont enregistrees d'un bloc dans la base de donnees MySQL.
