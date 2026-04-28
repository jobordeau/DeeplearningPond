# Pond — Apprentissage par renforcement

> Implémentation et comparaison de plusieurs variantes de Deep Q-Learning sur le jeu de plateau **Pond**, avec interface graphique pour observer les agents et affronter les modèles entraînés.

Projet réalisé dans le cadre du Mastère **Big Data & Intelligence Artificielle** à l'**ESGI** (matière : Apprentissage par renforcement). Le sujet imposait d'évaluer plusieurs algorithmes de RL sur un environnement personnalisé issu de [BoardGameArena](https://boardgamearena.com/gamepanel?game=pond).

---

## Sommaire

- [Aperçu](#aperçu)
- [Le jeu Pond](#le-jeu-pond)
- [Algorithmes implémentés](#algorithmes-implémentés)
- [Résultats](#résultats)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Détails techniques](#détails-techniques)

---

## Aperçu

Ce projet contient :

- Un **environnement Pond** entièrement réécrit en Python (grille 4×4, espace d'actions de taille 144 avec masque d'actions valides)
- **5 agents** : `random`, `dqn`, `dqn_target`, `dqn_er`, `dqn_per`
- Une **interface graphique Pygame** pour regarder un agent jouer ou affronter un modèle
- Une **CLI unifiée** (`main.py`) pour entraîner, évaluer, jouer et benchmarker les agents
- Un **modèle pré-entraîné** (`models/best_model.pth`) atteignant ~**87 % de winrate** contre un adversaire aléatoire

---

## Le jeu Pond

Pond est un jeu de plateau abstrait à deux joueurs (Light vs Dark) joué sur une grille 4×4. Chaque joueur dispose de 13 jetons et le but est d'atteindre 10 points avant l'adversaire.

### Pièces et évolution

Chaque jeton existe sous trois formes (un cycle d'évolution) :

| Pièce      | Symbole | Déplacement                  |
|------------|:-------:|------------------------------|
| Œuf        | E       | Aucun                        |
| Têtard     | T       | 1 case (haut/bas/gauche/droite) |
| Grenouille | F       | 1 ou 2 cases (haut/bas/gauche/droite) |

À chaque coup, les pièces **adjacentes à la case jouée** évoluent (Œuf → Têtard → Grenouille → Œuf).

### Marquer des points

Trois pièces du **même type** alignées (ligne ou colonne, 3 ou 4 d'affilée) sont retirées du plateau. Chaque joueur marque autant de points qu'il avait de pièces dans cette ligne.

### Fin de partie

- Un joueur atteint **10 points** → il gagne
- Un joueur ne peut plus jouer (élimination) → il perd

L'agent contrôle toujours **Light** et joue en premier. L'adversaire (Dark) joue de manière aléatoire pendant l'entraînement.

---

## Algorithmes implémentés

L'objectif pédagogique du sujet était d'évaluer la progression apportée par chaque amélioration du DQN classique. Les quatre variantes sont organisées en héritage progressif :

```
DQNAgent (vanilla)
    └── DQNTargetAgent (+ target network)
            ├── DQNExperienceReplayAgent (+ replay buffer)
            └── DQNPrioritizedReplayAgent (+ prioritized replay)
```

| Agent          | Target Network | Replay Buffer | Priorisation | Description                                                                 |
|----------------|:--------------:|:-------------:|:------------:|-----------------------------------------------------------------------------|
| `random`       | —              | —             | —            | Joue un coup légal au hasard (baseline)                                     |
| `dqn`          | ❌             | ❌            | ❌           | DQN vanilla : un seul réseau, mises à jour en ligne                         |
| `dqn_target`   | ✅             | ❌            | ❌           | Ajout d'un réseau cible synchronisé toutes les N étapes                     |
| `dqn_er`       | ✅             | ✅            | ❌           | Échantillonnage uniforme dans un buffer d'expériences (10 000 transitions)  |
| `dqn_per`      | ✅             | ✅            | ✅           | Buffer priorisé selon le TD-error \|δ\| (α=0.6, β annélé de 0.4 à 1.0)    |

### Choix d'architecture

- **Réseau** : MLP partagé entre tous les agents (`pond_rl/networks/q_network.py`) — 3 couches denses 256-256, ReLU, sortie de taille 144
- **Encodage de l'état** : vecteur de taille 69 = 16 cases × 4 dims (couleur + one-hot Œuf/Têtard/Grenouille) + 5 features globales (scores, jetons restants, joueur courant)
- **Action masking** : à chaque coup, l'environnement calcule un masque binaire des 144 actions légales. Les Q-values des actions interdites sont mises à `-inf` avant l'`argmax`. Indispensable car ~90 % des actions sont illégales à chaque état.

### Hyperparamètres par défaut

```python
hidden_dim         = 256
learning_rate      = 1e-3
gamma              = 0.99
epsilon (start)    = 1.0     epsilon_min = 0.1     epsilon_decay = 0.995
target_update_freq = 100     (DQN target / ER / PER)
batch_size         = 64      (ER / PER)
buffer_capacity    = 10000   (ER / PER)
warmup_steps       = 200     (ER / PER)
alpha              = 0.6     (PER)
beta_start         = 0.4     beta_increment = 1e-4 (PER)
```

---

## Résultats

Évaluation sur 200 parties contre un adversaire aléatoire (joueur Light, premier à jouer). Modèle de référence : `models/best_model.pth`.

| Agent        | Win %      | Lose %     | Tie %     | Longueur moy. | Temps/coup |
|--------------|-----------:|-----------:|----------:|--------------:|-----------:|
| `random`     |     52.5 % |     43.5 % |    4.0 %  |     20.4 step |   0.004 ms |
| `dqn_target` | **90.5 %** |  **9.0 %** |   0.5 %   |     15.2 step |   0.5 ms   |

L'agent entraîné gagne **+38 points de winrate** par rapport à un joueur aléatoire et clôt les parties **~5 coups plus tôt** en moyenne (parties plus efficaces, davantage de scoring lines créées rapidement).

> Pour reproduire : `python main.py evaluate --agent dqn_target --model models/best_model.pth --episodes 1000`

---

## Architecture du projet

```
DeeplearningPond/
├── main.py                          # CLI unifiée (train / evaluate / play / benchmark)
├── requirements.txt
├── .gitignore
├── README.md
├── models/
│   └── best_model.pth               # Modèle pré-entraîné (~87 % winrate)
├── docs/
│   └── encoding.pdf                 # Documentation sur l'encodage état/action
└── pond_rl/                         # Package principal
    ├── env/                         # Environnement & règles du jeu
    │   ├── pond_env.py
    │   ├── board.py
    │   ├── player.py
    │   └── token.py
    ├── agents/                      # Algorithmes d'apprentissage
    │   ├── base_agent.py
    │   ├── random_agent.py
    │   ├── dqn.py
    │   ├── dqn_target.py
    │   ├── dqn_experience_replay.py
    │   └── dqn_prioritized_replay.py
    ├── networks/
    │   └── q_network.py             # MLP partagé par tous les agents DQN
    ├── utils/
    │   ├── replay_buffer.py
    │   ├── prioritized_replay_buffer.py
    │   ├── model_io.py              # save / load / top-k tracking
    │   └── evaluation.py
    └── gui/
        ├── display.py               # Rendu Pygame
        └── play.py                  # Modes humain vs agent / spectateur
```

---

## Installation

### Prérequis

- **Python 3.10+** (testé sur 3.12)
- **GPU NVIDIA** optionnel (CUDA 12.x) — l'entraînement fonctionne aussi sur CPU

### Étapes

```bash
git clone https://github.com/<votre-utilisateur>/DeeplearningPond.git
cd DeeplearningPond

python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

`requirements.txt` :

```
torch>=2.0
numpy>=1.24
pygame>=2.5
tqdm>=4.65
```

> 💡 Pour une installation CPU plus légère, remplacer la première ligne par :
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Vérification

```bash
python main.py evaluate --agent dqn_target --model models/best_model.pth --episodes 200
```

Vous devriez voir un winrate autour de **86–90 %**.

---

## Utilisation

La CLI expose 4 commandes principales : `train`, `evaluate`, `play`, `benchmark`.

### Entraîner un agent

```bash
python main.py train --agent dqn_target --episodes 10000 --eval-interval 200
```

Options principales :

| Option              | Défaut | Description                                          |
|---------------------|:------:|------------------------------------------------------|
| `--agent`           | —      | `random`, `dqn`, `dqn_target`, `dqn_er`, `dqn_per`   |
| `--episodes`        | 10000  | Nombre d'épisodes d'entraînement                     |
| `--eval-interval`   | 200    | Fréquence d'évaluation (en épisodes)                 |
| `--eval-episodes`   | 100    | Nombre de parties par évaluation                     |
| `--save-folder`     | `models/<agent>` | Dossier de sauvegarde des checkpoints      |
| `--device`          | `auto` | `cpu`, `cuda`, ou `auto`                             |

Pendant l'entraînement, les **5 meilleurs modèles** (par winrate) sont conservés dans `models/<agent>/`. Les autres checkpoints sont automatiquement supprimés.

### Évaluer un modèle

```bash
python main.py evaluate --agent dqn_target --model models/best_model.pth --episodes 1000
```

Affiche : winrate, defeat rate, tie rate, longueur moyenne des parties, temps moyen par coup.

### Jouer en interface graphique

Quatre modes d'affichage :

```bash
python main.py play --mode human_vs_agent --agent dqn_target --model models/best_model.pth
python main.py play --mode human_vs_random
python main.py play --mode watch_agent --agent dqn_target --model models/best_model.pth
python main.py play --mode watch_random
```

| Mode               | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `human_vs_agent`   | Vous (Dark par défaut) jouez contre un agent entraîné        |
| `human_vs_random`  | Vous jouez contre un agent aléatoire                         |
| `watch_agent`      | Regarder l'agent entraîné jouer contre un agent aléatoire    |
| `watch_random`     | Random vs Random (utile pour observer la dynamique du jeu)   |

Contrôles : clic gauche pour placer un œuf sur une case vide, clic sur une de vos pièces puis sur la destination pour la déplacer.

### Benchmarker tous les agents

```bash
python main.py benchmark --episodes 500 --models-dir models
```

Lance une évaluation sur tous les agents enregistrés en chargeant automatiquement le meilleur modèle de chaque dossier `models/<agent>/`. Sortie sous forme de tableau comparatif.

---

## Détails techniques

### Encodage de l'état (69 dimensions)

```
indices 0..63   : 16 cases × 4 valeurs (color_bit, frog, tadpole, egg)
indices 64..68  : light_score, dark_score, light_tokens_left, dark_tokens_left, current_player
```

### Encodage des actions (144 actions)

Pour chaque case (16 au total), 9 actions possibles :

| ID local | Type        | Détail                          |
|---------:|-------------|---------------------------------|
|        0 | place       | Placer un œuf sur cette case    |
|      1–4 | move        | Déplacer 1 case (haut/D/bas/G) |
|      5–8 | move        | Déplacer 2 cases (haut/D/bas/G) |

Soit `action_id = case_id * 9 + sous-action`. Voir `docs/encoding.pdf` pour les schémas complets.

### Système de récompenses

```python
+ 0.1 par point marqué pendant la partie
+ 1.0 (+ delta de score) à la victoire
- 1.0 à la défaite
  0.0 en cas d'égalité
```

La récompense intermédiaire `+0.1` sur scoring permet de débloquer l'apprentissage rapidement (le signal binaire victoire/défaite est trop épars pour des parties de 20–30 coups).

### Méthodologie du choix d'hyperparamètres

- **`learning_rate` = 1e-3** : valeur classique pour Adam sur des MLPs de cette taille
- **`gamma` = 0.99** : les parties font ~20–30 coups, la récompense terminale doit propager loin
- **`epsilon_decay` = 0.995** : empiriquement, descend à `epsilon_min` en ~500 épisodes — un bon compromis
- **`target_update_freq` = 100 épisodes** : valeur recommandée par Mnih et al. pour des problèmes de cette échelle
- **`buffer_capacity` = 10 000** : couvre ~500 épisodes — suffisant pour décorréler sans noyer les transitions récentes
- **PER `alpha` = 0.6, `beta_start` = 0.4** : valeurs canoniques de Schaul et al. (2016)

---

## Auteur

Projet réalisé en M2 Big Data & Intelligence Artificielle à l'**ESGI**.

Encadrant : Nicolas VIDAL — `nvidal@myges.fr`

Référence pédagogique : *Reinforcement Learning: An Introduction* — Sutton & Barto.
