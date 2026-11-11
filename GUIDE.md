# GUIDE — Call Me Maybe (42)

> Guide de travail pour `cmm.pdf` v1.5. Il ne remplace pas le sujet : il te dit
> **dans quel ordre** construire, **pourquoi** chaque brique existe, et **comment
> vérifier** que tu as le droit de passer à la suivante.
>
> Règle du jeu (cf. `AGENTS.md`) : ce guide ne contient pas ton code métier.
> Chaque étape finit par des questions. Si tu ne peux pas répondre, ne code pas
> encore : relis, dessine, ou demande-moi de t'expliquer.

---

## Sommaire

1. [Le projet en une phrase](#1-le-projet-en-une-phrase)
2. [Ce que tu rends / comment tu es noté](#2-ce-que-tu-rends--comment-tu-es-noté)
3. [Les 6 concepts à maîtriser avant de coder](#3-les-6-concepts-à-maîtriser-avant-de-coder)
4. [Architecture cible](#4-architecture-cible)
5. [Étape 0 — Environnement (déjà initialisé)](#étape-0--environnement-déjà-initialisé)
6. [Étape 1 — Entrées + pydantic](#étape-1--entrées--pydantic)
7. [Étape 2 — Parler au modèle (baseline naïve)](#étape-2--parler-au-modèle-baseline-naïve)
8. [Étape 3 — Vocabulaire : token → caractères](#étape-3--vocabulaire--token--caractères)
9. [Étape 4 — La grammaire (automate du JSON)](#étape-4--la-grammaire-automate-du-json)
10. [Étape 5 — Masquage des logits (token par token)](#étape-5--masquage-des-logits-token-par-token)
11. [Étape 6 — La boucle de génération](#étape-6--la-boucle-de-génération)
12. [Étape 7 — Sortie et validation](#étape-7--sortie-et-validation)
13. [Étape 8 — Robustesse (le sujet y tient beaucoup)](#étape-8--robustesse-le-sujet-y-tient-beaucoup)
14. [Étape 9 — Performance](#étape-9--performance)
15. [Étape 10 — Tests, lint, README](#étape-10--tests-lint-readme)
16. [Grille d'auto-évaluation](#grille-dauto-évaluation)
17. [Pièges classiques](#pièges-classiques)
18. [Ressources](#ressources)

---

## 1. Le projet en une phrase

> Un modèle de langage de 0,6 milliard de paramètres ne sait pas produire du JSON
> fiable (≈30 % de réussite). On va le **forcer** à n'écrire que du JSON valide,
> caractère après caractère, en filtrant ses logits : c'est le **constrained
> decoding**.

Exemple concret :

```
"Quelle est la somme de 40 et 2 ?"
        ↓  (ton programme, pas le modèle seul)
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

Le programme **ne calcule pas 42**. Il choisit *quelle fonction appeler* et
*avec quels arguments typés*.

### Pourquoi c'est intéressant

| Sans contrainte | Avec contrainte |
|---|---|
| le modèle écrit `Sure! The answer is {"a": 40` | le modèle ne *peut pas* écrire autre chose qu'un JSON conforme |
| JSON invalide 1 fois sur 3 | JSON valide 100 % (structure garantie par toi, pas par lui) |
| il faut parser, réparer, réessayer | `json.loads` réussit toujours |

**La nuance importante** : tu garantis la *forme*, pas la *justesse*. Choisir
`fn_greet` au lieu de `fn_add_numbers` reste une erreur du modèle. D'où
l'exigence « 90 %+ de bonnes sélections » : ça se joue sur le **prompt** et sur
la **qualité des logits**, pas sur la grammaire.

---

## 2. Ce que tu rends / comment tu es noté

Le dépôt doit contenir :

```
cmm/
├── src/                     # ton code
├── llm_sdk/                 # fourni (déjà en place)
├── data/input/              # fourni (déjà en place)
├── pyproject.toml + uv.lock # déjà en place
├── Makefile                 # déjà en place
├── README.md                # à écrire (chapitre VI du sujet)
└── .gitignore               # déjà en place
```

`data/output/` **ne doit pas** être commité (il est déjà dans `.gitignore`).

Commandes que la moulinette lance :

```sh
make install     # = uv sync
make run         # = uv run python -m src
make lint        # flake8 + mypy avec les flags obligatoires
make debug       # = pdb
```

Le jour J, la moulinette remplace `data/input/*.json` par d'autres fichiers.
**Tout ce qui marche seulement avec les exemples fournis est faux.**

---

## 3. Les 6 concepts à maîtriser avant de coder

### 3.1 Token

Un modèle ne voit pas des lettres : il voit des **tokens** (sous-mots). Ton
implémentation raisonne donc en deux alphabets différents :

| Alphabet | Unité | Qui le manipule |
|---|---|---|
| caractères | `"`, `{`, `4`, `a`… | ta grammaire JSON |
| tokens | `Ġ{`, `name`, `": "`… | le modèle (logits) |

La difficulté centrale du projet est la **traduction entre les deux**, dans les
deux sens : logits (par token) → probabilité, puis token choisi → texte.

> Question : si le modèle propose un token qui vaut `"},"`, combien de
> transitions de ta grammaire consommes-tu d'un coup ?

### 3.2 Logits

`get_logits_from_input_ids(ids)` renvoie **un score par token du vocabulaire**
(≈151 000 nombres). Ce ne sont pas des probabilités : pas de somme à 1, pas
bornées. Prendre `argmax` des logits = choisir le token le plus probable
(décodage *greedy*, déterministe).

> Question : pourquoi mettre `-inf` sur un logit interdit suffit-il à
> l'éliminer, alors que mettre `0` ne suffirait pas ?

### 3.3 Vocabulaire

`get_path_to_vocab_file()` donne un `vocab.json` de la forme
`{"token": id, ...}`. Attention : Qwen utilise un BPE **byte-level** (comme
GPT-2). Les tokens ne sont pas du texte brut. Deux surprises garanties :

1. l'espace est encodé `Ġ` (ce n'est pas un « G » pointé, c'est un caractère
   spécial de la table) ;
2. les octets non imprimables sont projetés sur des caractères Unicode
   arbitraires (`Ã©` peut vouloir dire `é`).

> Question : si un token contient `\xc3\xa9`, c'est-à-dire les **deux octets**
> UTF-8 de `é`, comment reconstruis-tu la chaîne `"é"` ?

C'est l'indice du sujet : « *Think about how you can use the vocabulary JSON
file to map between tokens and their string representations* ». Il faut
reconstruire la table inverse `id → bytes` avec la même bijection
octet → caractère que celle du BPE.

### 3.4 JSON et schéma

Deux JSON différents, ne pas les confondre :

| Fichier | Rôle | Contenu |
|---|---|---|
| `functions_definition.json` | *ce que le programme sait faire* | nom, description, paramètres typés, type de retour |
| `function_calling_tests.json` | *les questions* | `[{"prompt": "..."}]` |

Types possibles pour un paramètre : `number`, `string`, `boolean` (le sujet dit
« etc. », donc prévois l'inconnu en erreur propre plutôt qu'en crash).

### 3.5 Constrained decoding

À chaque token :

1. le modèle produit les logits ;
2. tu calcules l'ensemble des tokens **légaux** à cette position ;
3. tu mets `-inf` partout ailleurs ;
4. tu prends `argmax` ;
5. tu ajoutes le token choisi à l'entrée, et tu recommences.

> Question : à la position 0, quels tokens sont légaux ? Combien ? Et après
> `{"name": "fn_a` ?

### 3.6 Pydantic

Toutes les classes doivent en hériter (exigence IV.3.1). C'est aussi ton
meilleur outil pour valider les entrées : un fichier mal formé doit être rejeté
proprement, pas provoquer un `KeyError`.

---

## 4. Architecture cible

Découpage conseillé (adapte les noms, mais garde la séparation des
responsabilités) :

| Fichier | Responsabilité | Dépend de |
|---|---|---|
| `src/__main__.py` | CLI `argparse`, orchestration, codes de sortie | tout |
| `src/models.py` | schémas pydantic (définitions, appels, sortie) | — |
| `src/loader.py` | lire + valider les deux JSON d'entrée | `models` |
| `src/prompt.py` | construire le texte envoyé au modèle | `models` |
| `src/vocab.py` | `vocab.json` → `id → str`, filtrage des tokens spéciaux | — |
| `src/grammar.py` | automate du JSON attendu (états, transitions) | `models` |
| `src/mask.py` | tokens autorisés par état (+ cache / trie) | `vocab`, `grammar` |
| `src/decoder.py` | boucle de génération contrainte | `mask`, `grammar` |
| `src/writer.py` | écrire la sortie (crée `data/output/`) | `models` |

**Deux décisions de design à prendre consciemment** (elles seront demandées en
soutenance) :

1. **Faut-il générer le champ `prompt` ?** Non : il vient de l'entrée, tu le
   recopies. Générer seulement `name` + `parameters` réduit le nombre de tokens.
2. **Le `decoder` doit-il recevoir le modèle en paramètre ?** Oui :
   `decode(logits_fn, ...)` avec `logits_fn: Callable[[list[int]], list[float]]`.
   Tu pourras alors tester toute la mécanique **sans** charger 0,6 B de
   paramètres. C'est aussi ce qui rend une « modification en soutenance »
   faisable en 3 minutes.

---

## Étape 0 — Environnement (déjà initialisé)

Ce qui a été mis en place pour toi :

| Fichier | Ce qu'il fait |
|---|---|
| `pyproject.toml` | deps `numpy`, `pydantic`, `llm-sdk` (dépendance locale vers `./llm_sdk`) |
| `[tool.uv.sources]` | `llm-sdk = { path = "llm_sdk", editable = true }` → `import llm_sdk` marche après `uv sync` |
| `[[tool.uv.index]]` | torch **CPU only** (évite ~3 Go de wheels CUDA inutiles) |
| `Makefile` | `install`, `run`, `debug`, `clean`, `lint`, `lint-strict`, `test` |
| `.gitignore` | exclut `.venv`, caches, `data/output/` |

Vérifie :

```sh
make install
uv run python -c "from llm_sdk import Small_LLM_Model; print('ok')"
uv run python -c "import torch; print(torch.__version__)"
```

> **Pourquoi `llm-sdk` en dépendance locale ?** Parce que « copier le dossier à
> côté de `src` » ne suffit pas : `llm_sdk/` contient `llm_sdk/llm_sdk/__init__.py`.
> Sans installation, `import llm_sdk` tombe sur un *namespace package* vide.
> Avec `uv sync`, le paquet est installé dans `.venv` et l'import est fiable quel
> que soit le répertoire courant.

> **Pourquoi torch CPU ?** 6 cœurs, pas de GPU, 7 Go de RAM. Les wheels CUDA
> pèsent ~3 Go et ne servent à rien ici.

Premier lancement du modèle : il télécharge `Qwen/Qwen3-0.6B` depuis Hugging
Face (~1,2 Go) dans `~/.cache/huggingface`. Ça ne se produit qu'une fois.

> **Spécifique à cette machine** : `/home` fait 4,7 Go et était déjà à 90 %.
> J'ai donc redirigé le cache uv vers `/sgoinfre/students/durisosa/.cache/uv`
> via `~/.config/uv/uv.toml`. Pour le cache du modèle, ajoute dans ton
> `~/.bashrc` :
>
> ```sh
> export HF_HOME=/sgoinfre/students/durisosa/.cache/huggingface
> ```
>
> Sans ça, le téléchargement du modèle peut échouer par manque d'espace.
> Ce réglage est **local** : il ne part pas dans le dépôt et ne gêne pas
> l'évaluateur.

Mesures déjà faites sur cette machine (elles te donnent un ordre de grandeur) :

| Mesure | Valeur |
|---|---|
| chargement du modèle, à froid (lecture NFS) | 133 s |
| chargement du modèle, à chaud | 15 s |
| forward sur un prompt de 79 tokens | 0,40 s |
| forward steady-state (prompt court) | 0,13 s |
| 10 tokens greedy sur un prompt de 79 tokens | 3,42 s |
| `encode("Hello")` | **1** token |
| entrées de `vocab.json` | 151 643 |
| dimensions des logits renvoyées par le SDK | **151 936** |

Conclusion : ~0,35 s/token, donc ~30 tokens par réponse ⇒ ~2 min pour 11
prompts. C'est jouable, mais **mesure et documente**, et surveille le nombre de
tokens générés.

> **Pourquoi la dépendance `torch` est listée explicitement ?** `uv` n'applique
> `[tool.uv.sources]` qu'aux dépendances **directes**. Sans cette ligne, le
> `torch` tiré par `llm-sdk` venait de PyPI avec les wheels CUDA (3 Go) et le
> lock contenait 36 paquets `nvidia-*`. La ligne + l'index `pytorch-cpu`
> garantissent `torch==2.14.1+cpu` (vérifie : `grep -c 'name = "nvidia' uv.lock`
> doit renvoyer `0`).

Questions de sortie d'étape :

- Que fait exactement `uv sync` ? Que contient `uv.lock` et pourquoi le
  commiter ?
- Pourquoi le sujet interdit-il `torch`/`transformers` **dans ton code** alors
  que `llm_sdk` les utilise ?

---

## Étape 1 — Entrées + pydantic

Objectif : charger les deux fichiers, les valider, et afficher un résumé.

À faire :

1. Modéliser une définition de fonction (`name`, `description`, `parameters`,
   `returns`), un paramètre (`type`), un prompt d'entrée.
2. Valider que `name` est non vide, que `parameters` est un dict, que chaque
   `type` est connu.
3. Refuser proprement : fichier absent, JSON invalide, racine qui n'est pas une
   liste, liste vide, doublon de nom de fonction.

Critère de validation :

```sh
uv run python -m src --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json --output /tmp/out.json
# doit afficher 5 fonctions / 11 prompts, sans rien écrire encore
```

Questions :

- Que doit-il se passer si `data/input/` n'existe pas ? Et si `--output` pointe
  vers un dossier inexistant ?
- Un `prompt` vide est-il une erreur d'entrée ou un cas normal à traiter ?

---

## Étape 2 — Parler au modèle (baseline naïve)

**Ne saute pas cette étape.** Elle te donne la preuve chiffrée que la
contrainte est nécessaire — et un chiffre à mettre dans ton README.

À faire :

1. Construire un prompt qui liste les fonctions (nom + description + paramètres)
   et la question de l'utilisateur.
2. Générer ~80 tokens **sans aucune contrainte** (`argmax` sur les logits).
3. Compter combien de sorties sont du JSON parsable et conforme.

Format de prompt pour Qwen3 (chat template, sans `encode` de template — tu
écris le texte toi-même) :

```
<|im_start|>system
Tu es un assistant qui répond uniquement par un appel de fonction JSON.
<|im_end|>
<|im_start|>user
<définitions>
Question: <prompt>
<|im_end|>
<|im_start|>assistant
```

Questions :

- Combien de sorties sur 11 sont valides ? Note le chiffre.
- Le modèle émet-il un bloc ` thinking` ? Comment l'éviter ou le rendre inoffensif ?
- Combien de secondes par token sur cette machine ? (mesure : `time.perf_counter`
  autour de chaque appel `get_logits_from_input_ids`)
- Que renvoie `encode()` exactement, et comment le passer à
  `get_logits_from_input_ids` (qui attend une `list[int]`) ?

> Piège : à chaque token il faut **renvoyer toute la séquence** au modèle.
> C'est O(n²) sur la longueur, mais le SDK n'expose pas de cache KV. Sur 40
> tokens c'est acceptable ; mesure avant d'optimiser.

---

## Étape 3 — Vocabulaire : token → caractères

Objectif : une fonction `token_text(token_id) -> bytes | None`.

À faire :

1. Lire `vocab.json` (chemin via `get_path_to_vocab_file()`).
2. Construire la bijection inverse du byte-level BPE.
3. Exclure les tokens spéciaux (ceux qui ne décodent pas en octets normaux).
4. Vérifier sur des cas concrets.

Test de validation (indispensable) :

| Token du fichier | Ce que tu dois obtenir |
|---|---|
| `Ġ{` | `b" {"` |
| `name` | `b"name"` |
| `\u00c3\u00a9` (deux caractères) | `b"\xc3\xa9"`, soit `"é"` |
| `\u0120` | `b" "` |

Vérification globale : pour **tous** les ids, `decode(token_text(id))` doit
redonner exactement le token d'origine. Si un seul diffère, ta bijection est
fausse.

> **Piège mesuré sur ce modèle** : `vocab.json` contient **151 643** entrées
> (ids 0 → 151 642), mais `get_logits_from_input_ids` renvoie **151 936**
> valeurs : le vocabulaire du réseau est *paddé*. Les ~293 ids en trop n'ont
> **aucun texte**. Si tu ne les masques pas explicitement, ton `argmax` peut
> tomber dessus et tu auras un token indécodable. Autrement dit : tout id
> absent de `vocab.json` est **illégal par définition**.
>
> Autre piège : `Ġ` est bien le caractère `U+0120` (et non un « G »), et le
> vocabulaire contient des séquences de 1, 2 voire 3 caractères apparents qui
> représentent 1 à 3 **octets** UTF-8.

Questions :

- Pourquoi travailler en **octets** et pas en `str` ? (Indice : découpage UTF-8
  au milieu d'un caractère.)
- Un token peut-il produire 0 octet ? Que faire de lui ?
- Combien de tokens contiennent un `"` ou un `\` ? Cela va-t-il poser problème
  pour les chaînes JSON ?
- Que fais-tu des 293 ids sans texte : exception, ou exclusion silencieuse ?

---

## Étape 4 — La grammaire (automate du JSON)

Objectif : un automate à états finis qui dit, pour un préfixe donné, si l'on
peut encore atteindre un JSON valide pour **ce** schéma.

Forme de sortie que tu dois produire (canonique, sans espaces inutiles) :

```
{"name": "<un des noms>", "parameters": {"<clef>": <valeur>, ...}}
```

Décompose en familles d'états :

| État | Autorise ensuite |
|---|---|
| `START` | `{` |
| `KEY_NAME` (littéral) | `"name"` caractère par caractère |
| `NAME_VALUE` | `"` puis les caractères d'**un nom de fonction connu** |
| `PARAMS_OPEN` | `"parameters": {` |
| `PARAM_KEY` | une clef attendue pour cette fonction (ou `}` si aucune) |
| `PARAM_VALUE_NUMBER` | `-`, chiffre, `.`, `e`… |
| `PARAM_VALUE_STRING` | tout caractère **hors** `"` et `\` (ou gère les échappements) |
| `PARAM_VALUE_BOOL` | `true` / `false` |
| `AFTER_VALUE` | `,` (autre paramètre) ou `}` (fin des paramètres) |
| `END` | `}` puis **stop** |

Points de conception à trancher :

1. **L'énumération des noms** : c'est *le* levier du projet. À l'état
   `NAME_VALUE`, seuls les caractères des noms de fonctions fournis sont
   légaux. Le modèle ne peut pas inventer une fonction. Note dans ton README
   que la sélection reste faite **par le modèle** (argmax sur les tokens
   légaux) : ce n'est pas une heuristique.
2. **L'ordre des paramètres** : le schéma JSON ne l'impose pas. Le plus simple
   est de fixer *ton* ordre canonique (celui de `functions_definition.json`) et
   de l'exiger dans la grammaire. C'est légal (les clefs restent exactes) et ça
   réduit les états.
3. **Les nombres** : une seule machine pour `number` (entier **ou** flottant) :
   `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`. Attention à
   `01` (interdit en JSON), `1.` (incomplet), `-` seul (incomplet).
4. **Les chaînes** : si tu autorises uniquement les caractères « sûrs », tu
   refuseras des réponses légitimes (une question peut contenir `"`). Deux
   options : gérer `\"`, `\\`, `\n`, `\uXXXX` dans la grammaire, ou assumer la
   limitation et la documenter. **Choisis et justifie.**
5. **La fin** : l'état acceptant doit être détecté *avant* de générer un token
   de trop. Il n'y a pas de token EOS à attendre.

Critère de validation (à écrire comme test, voir étape 10) :

- toute chaîne acceptée par l'automate ⇒ `json.loads` réussit **et** valide le
  schéma pydantic ;
- `{"name": "fn_", ...}` avec un nom incomplet ⇒ refusé ;
- `{"name": "fn_add_numbers", "parameters": {"a": 1, "b": 2}, "x": 0}` ⇒ refusé
  (pas de clef en trop).

Questions :

- Combien d'états au total, pour 5 fonctions et 3 paramètres maximum ? Et si le
  fichier d'entrée contient 40 fonctions ?
- Pourquoi un automate à états est-il préférable à « générer puis réparer le
  JSON » ?

---

## Étape 5 — Masquage des logits (token par token)

Objectif : `allowed_ids(state) -> collection d'ids`.

Algorithme naïf (à écrire d'abord, même s'il est lent) :

> pour chaque id du vocabulaire : prendre son texte, le faire consommer par
> l'automate depuis l'état courant ; s'il est accepté, garder l'id.

C'est correct, et **trop lent** : 151 000 tokens × ~40 tokens × 11 prompts
≈ 66 millions de simulations. Mesure-le, puis optimise.

Deux optimisations, à faire dans cet ordre :

1. **Cache par état** : `dict[state, list[int]]`. Le nombre d'états est petit
   et indépendant de la longueur générée (les états « contenu de chaîne » et
   « contenu de nombre » sont génériques). Tu payes le calcul une fois par
   état, pas une fois par token généré.
2. **Trie des tokens** (ou filtrage par classe de caractères) : au lieu de
   tester chaque token indépendamment, parcours un arbre préfixe des textes de
   tokens ; dès qu'un caractère est refusé, tu élagues toute la branche. Pour
   les états « contenu de chaîne » (immense majorité des tokens), un simple
   précalcul « tokens dont tous les octets ∈ classe autorisée » suffit.

Critère de validation :

- le premier token proposé est toujours `{` ;
- après `{"name": "fn_` le token `fn_add_numbers` complet n'est proposé que
  s'il permet d'atteindre un état valide ;
- aucun token autorisé ne peut faire sortir de la grammaire ;
- temps de calcul du masque < ~5 ms par token après cache.

Questions :

- Que se passe-t-il si **aucun** token n'est légal (bug de grammaire) ?
  Comment le détecter au lieu de planter ?
- `argmax` sur une liste où tout vaut `-inf` renvoie 0. Pourquoi est-ce un piège ?
- Faut-il autoriser les espaces après `:` et `,` ? Qu'est-ce que ça change pour
  ton cache ?

---

## Étape 6 — La boucle de génération

Pseudo-déroulé (à traduire en Python, pas à copier) :

```
ids   = encode(prompt)
state = START
out   = b""
tant que state n'est pas final :
    logits = logits_fn(ids)              # 151k floats
    ids_ok = allowed_ids(state)
    si ids_ok est vide -> erreur explicite
    masque = -inf partout ; logits[id] pour id dans ids_ok
    best   = argmax(masque)
    texte  = token_text(best)
    si texte est None -> erreur (token spécial autorisé par erreur)
    state  = consume(state, texte)       # ou échec => bug
    out   += texte
    ids   += [best]
json.loads(out)
```

Détails qui comptent :

- **Greedy obligatoire** (pas d'échantillonnage) : sinon un token légal mais
  improbable casse la reproductibilité. Note-le dans le README.
- **`json.loads` après génération** : c'est ton filet de sécurité. Si ça lève,
  c'est un bug de grammaire, pas du modèle. Traite-le comme une erreur interne
  claire.
- **Compteur de tokens max** : garde-fou anti-boucle infinie (ex. 256). S'il est
  atteint, erreur explicite.
- **Le prompt ne doit pas contenir de token qui désaligne la grammaire** : tu
  ne génères que la partie JSON, jamais du texte libre.

Questions :

- Pourquoi réencoder toute la séquence est-il correct, mais sous-optimal ?
- Où placer le `try/except` : autour de chaque prompt, ou autour de tout le
  fichier ? Que doit produire une erreur sur le prompt n° 7 ?
- Quelle est la différence entre « le modèle s'est trompé de fonction » et
  « ton programme a produit un JSON invalide » ? Un seul des deux est grave.

---

## Étape 7 — Sortie et validation

Format exact (trois clefs, rien d'autre) :

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

À faire :

1. Créer `data/output/` si absent (`Path.mkdir(parents=True, exist_ok=True)`).
2. Écrire en UTF-8, JSON indenté, `ensure_ascii=False` (ou `True`, mais sois
   cohérent et teste les accents).
3. Écrire **atomiquement** si tu veux être propre : fichier temporaire puis
   `os.replace` — évite un fichier à moitié écrit si ça plante.
4. Re-valider la sortie avec tes modèles pydantic avant d'écrire.

Vérification :

```sh
uv run python -m json.tool data/output/function_calling_results.json > /dev/null \
  && echo "JSON valide"
```

Questions :

- `{"a": 2}` (int) est-il conforme pour un paramètre `number` ? Et `{"a": true}` ?
  (Rappel : en Python `True == 1` ; en JSON `true` n'est pas un nombre.)
- Que faire si la fonction choisie n'a **aucun** paramètre ? Comment la
  grammaire produit-elle `{}` ?
- Le sujet exige « exactement les clefs prompt/name/parameters ». Un `sort_keys`
  change-t-il la validité ?

---

## Étape 8 — Robustesse (le sujet y tient beaucoup)

Le sujet : *« Your program must never crash unexpectedly and must always
provide clear error messages »*. Tableau des cas à couvrir :

| Cas | Comportement attendu |
|---|---|
| fichier d'entrée absent | message clair sur `stderr`, code de sortie ≠ 0 |
| JSON invalide | message + position si possible, pas de traceback |
| racine non-liste / liste vide | message clair ; liste vide ⇒ sortie `[]` |
| définition sans `name` | erreur de validation pydantic |
| type de paramètre inconnu | erreur explicite, pas de `KeyError` |
| prompt non-str | erreur de validation |
| dossier de sortie impossible à créer | message clair |
| modèle introuvable / pas de réseau | message clair + suggestion |
| `KeyboardInterrupt` | sortie propre |

Règles :

- **Pas de `print` de debug** dans le rendu final : utilise `logging` ou un
  `--verbose`.
- Les erreurs vont sur `stderr`, le résultat sur `stdout`.
- Un `except Exception` global **uniquement** au niveau `main`, pour convertir
  en message + code de sortie. Pas de `except: pass`.

Questions :

- Faut-il continuer après une erreur sur un prompt (et produire un élément
  `null`) ou s'arrêter ? Le sujet interdit les clefs en trop : que fait un
  élément `null` dans le tableau ?
- Comment distinguer « erreur utilisateur » (code 1) de « bug interne » (code 2) ?

---

## Étape 9 — Performance

Objectif du sujet : **< 5 minutes** pour tous les prompts.

Mesure d'abord, optimise ensuite. Points chauds probables, par ordre :

1. `get_logits_from_input_ids` (le forward du modèle) : incompressible sans
   cache KV. Réduire le **prompt** (descriptions courtes) et le nombre de
   tokens générés aide directement.
2. construction du masque : cache par état (étape 5).
3. `encode` répété : encode le prompt une seule fois, puis ajoute les ids.
4. conversions `list`/`numpy` : `np.asarray(logits)` + masque vectorisé.

Questions :

- Combien de tokens génère en moyenne une réponse ? Où pourrait-on en économiser
  (ordre des paramètres, pas d'espaces, `name` avant `parameters`) ?
- Pourquoi ne peut-on pas batcher plusieurs prompts en un seul appel au modèle ?
- Le temps de chargement du modèle compte-t-il dans les 5 minutes ? (Documente
  ta mesure : à froid / à chaud.)

---

## Étape 10 — Tests, lint, README

### Tests (`tests/`, non notés mais indispensables)

| Test | Ce qu'il vérifie | Coût |
|---|---|---|
| grammaire | toute chaîne acceptée parse et valide le schéma | instantané |
| rejets | `01`, `1.`, `-`, nom inconnu, clef en trop | instantané |
| vocab | `decode(token_text(id)) == token` pour tous les ids | ~1 s |
| decoder | avec un `logits_fn` factice, la sortie est toujours valide | instantané |
| loader | fichiers absents / malformés | instantané |
| intégration | 1 prompt réel avec le vrai modèle | lent (à marquer) |

Le test « decoder avec un faux `logits_fn` » est le plus rentable : il teste
toute la mécanique sans modèle.

### Lint

```sh
make lint          # flake8 . + mypy (flags obligatoires du sujet)
make lint-strict   # optionnel, mais bonne préparation à la soutenance
```

Rappels : max 79 colonnes (`.flake8`), docstrings PEP 257, type hints partout,
y compris les retours de `__init__` (`-> None`).

### README (chapitre VI — en anglais)

Première ligne **obligatoire**, en italique :

```markdown
*This project has been created as part of the 42 curriculum by <login>.*
```

Puis : Description, Instructions, Resources (+ usage de l'IA), et les sections
exigées : *Algorithm explanation*, *Design decisions*, *Performance analysis*,
*Challenges faced*, *Testing strategy*, *Example usage*.

---

## Grille d'auto-évaluation

Coche seulement ce que tu peux **prouver** :

- [ ] `make install` fonctionne sur un clone neuf
- [ ] `make run` produit `data/output/function_calling_results.json`
- [ ] le JSON de sortie est valide, sans clef en trop
- [ ] les types respectent `functions_definition.json` (number/string/boolean)
- [ ] 100 % des sorties parsables (testé sur ≥ 20 prompts)
- [ ] ≥ 90 % de bonnes fonctions (mesuré, chiffre dans le README)
- [ ] < 5 minutes au total, mesuré
- [ ] aucun crash sur les 9 cas d'erreur de l'étape 8
- [ ] `make lint` passe sans erreur
- [ ] aucune utilisation de `torch`/`transformers` dans `src/`
- [ ] aucun attribut privé du SDK (`_tokenizer`, `_model`, …)
- [ ] aucun nom de fonction ni valeur hardcodés
- [ ] README en anglais avec les 6 sections exigées
- [ ] `data/output/` absent du dépôt
- [ ] je sais expliquer chaque ligne de mon code à un pair

---

## Pièges classiques

| Piège | Symptôme | Correctif |
|---|---|---|
| `Ġ` pris pour un `G` | les tokens avec espace ne matchent jamais | décoder la table byte-level |
| comparaison en `str` au lieu de `bytes` | échec silencieux sur les accents | travailler en octets |
| `-inf` remplacé par `0` | tokens interdits réapparaissent | vraiment `-inf` |
| tout vaut `-inf` | `argmax` renvoie 0, sortie absurde | vérifier `ids_ok` non vide |
| grammaire qui accepte un préfixe invalide | `json.loads` échoue à la fin | tester l'automate en isolation |
| tokens spéciaux autorisés | `<|im_end|>` dans la sortie | filtrer dès l'étape 3 |
| logits paddés (151 936 vs 151 643) | token sans texte, sortie corrompue | masquer les ids absents du vocab |
| `max()` python sur 151k floats | lent (~ms perdu par token) | `numpy.argmax` |
| cache indexé par (état, position) | aucune réutilisation | indexer par **état seul** |
| ordre des paramètres non fixé | plusieurs JSON valides, masque incohérent | ordre canonique imposé |
| `True == 1` en Python | un booléen passe pour un nombre | valider sur le JSON, pas sur Python |
| test avec un seul run | « ça marche » puis échec en soutenance | répéter 5×, tester d'autres prompts |

---

## Ressources

| Sujet | Référence |
|---|---|
| Function calling | OpenAI, *Function calling* — guide officiel |
| Constrained decoding | *Efficient Guided Generation for LLMs* (Willard & Louf, 2023) |
| Grammaires + FSM | `outlines` (à lire, **interdit à installer**) |
| BPE byte-level | documentation GPT-2 / `tokenizers` |
| Qwen3 | fiche modèle Hugging Face `Qwen/Qwen3-0.6B` |
| JSON | RFC 8259 (grammaire des nombres et des chaînes) |
| Pydantic | documentation officielle v2 |
| Threading/typage mypy | documentation mypy, section `--strict` |
