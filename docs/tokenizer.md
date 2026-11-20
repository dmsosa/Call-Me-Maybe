# Construire `PersonalTokenizer`

Recette de construction, dans l'ordre. Chaque étape a son test dans
`tests/test_tokenizer.py` : ne passe pas à la suivante avant que les
siens soient verts.

## Contrat

| Méthode | Entrée | Sortie | Vérifié par |
|---|---|---|---|
| `__init__(vocab_path, merges_path)` | 2 chemins | objet prêt | `test_the_module_imports_and_exposes_encode_decode` |
| `encode(text)` | `str` | `list[int]` | 3 tests synthétiques + 5 d'intégration |
| `decode(ids)` | `list[int]` | `str` | `test_synthetic_round_trip`, `test_decode_matches_the_reference` |

Interdit : importer `llm_sdk`, appeler `model.encode`, importer
`tokenizers` ou `tiktoken`. Autorisé : `json`, `unicodedata`, `bytes`,
`regex` (déjà installé via `transformers`).

## Pipeline

```
texte
 1. NFC                      unicodedata.normalize
 2. pré-tokenisation         regex de tokenizer.json  -> morceaux
 3. octets                   morceau.encode("utf-8")
 4. octets -> caractères     table GPT-2 (espace -> Ġ)
 5. fusions BPE              merges.txt, par rang
 6. lookup                   vocab.json : token -> id
```

`decode` refait 6 → 5 → 4 → 3 en sens inverse, puis `bytes.decode`.

---

## J1 — la table octets ↔ caractères

Le fichier `vocab.json` est du texte : ni espace, ni `\n`, ni octet de
contrôle. GPT-2 a donc défini une bijection octet ↔ caractère imprimable :
188 octets gardés tels quels (ASCII imprimable + deux plages Latin-1),
les 68 autres décalés dans `U+0100+`, par ordre croissant d'octet.

**Invariants**

- les 256 entrées de la table sont une permutation de `0..255` ;
- `char_to_byte` est l'exacte inverse de `byte_to_char` ;
- `0x20` (espace) → `Ġ` (U+0120), `0x0A` → `Ċ` (U+010A) ;
- **le décalage dépend du rang d'exclusion, pas de l'octet** :
  `0x7F` → U+0121 (`ġ`), et non `256 + 127`.

**Vérifications avant de continuer**

1. les 256 tokens de base du `vocab.json` doivent occuper les ids `0..255` ;
2. `char_to_byte["Ġ"]` vaut 32 ;
3. l'aller-retour octet → caractère → octet est l'identité sur `0..255`.

Tu as déjà cette logique, correcte, dans `Vocab._build_char_to_byte`
(`src/core/vocab.py`). Le tokenizer a besoin des **deux** sens.

---

## J2 — NFC puis pré-tokenisation

**Invariants**

- l'entrée est normalisée en **NFC** avant tout découpage (jamais `.lower()`,
  jamais de suppression d'accents) ;
- la concaténation des morceaux redonne le texte normalisé : rien n'est
  perdu, rien n'est ajouté ;
- les morceaux ne dépendent pas de la longueur totale du texte ;
- l'espace reste attaché à ce qui le suit : `"a b"` → `["a", "Ġb"]` ;
- chaque chiffre est un morceau : `"-12.5"` → `["-", "1", "2", ".", "5"]`.

**Où trouver le motif** : `tokenizer.json`, clé `pre_tokenizer` →
`pretokenizers[0].pattern.Regex`. C'est une donnée du modèle, pas un choix.
À lire alternative par alternative (6 branches, séparées par `|`).
Le module `re` ne connaît pas `\p{L}`/`\p{N}` : d'où `regex`.

**Pièges** : appliquer le motif une seule fois (il faut toutes les
occurrences), oublier que l'ordre des alternatives compte, oublier NFC.

---

## J3 — `_bpe(piece)` : le cœur

Entrée : un morceau déjà passé par la table octets (chaîne de caractères
« sûrs »). Sortie : la liste des tokens fusionnés.

**Invariants**

1. on part des caractères du morceau, un par un ;
2. à chaque tour, on cherche la paire **de rang minimal** présente dans le
   morceau, puis on fusionne cette seule paire ;
3. après une fusion, tout est recalculé (la liste a changé) ;
4. si plusieurs occurrences ont le même rang, on prend la plus à gauche
   (comparer `(rang, indice)`) ;
5. on s'arrête quand aucune paire n'a de rang — **pas** quand un token du
   vocabulaire existe (voir `Greet` plus bas).

**Ne jamais** utiliser `id = 256 + rang` : c'est vrai pour ce
`vocab.json` (vérifié sur 151 387 fusions) mais c'est une propriété du
fichier, pas de l'algorithme. L'id vient de `tokens_to_ids[token]`.

**Le cas qui prouve que ce n'est pas un lookup** — `"Greet"` :

| passe | paire | rang | état |
|---|---|---|---|
| 1 | `r`+`e` | 9 | `G re e t` |
| 2 | `e`+`t` | 39 | `G re et` |
| 3 | `re`+`et` | 3488 | `G reet` |
| 4 | — | — | stop → `["G", "reet"]` |

Le merge `G`+`re` existe (rang 64 491) et donne le token `Gre` (id 64747),
mais il ne peut plus s'appliquer : `re` a été consommé. D'où
`[38, 3744]` et **non** `[64747, 295]`.

**Coût** : O(m²) consultations de dictionnaire, m = caractères du morceau
(quelques octets). Mesuré : 0,02 ms par prompt. Acceptable, à justifier
dans le README.

---

## J4 — `decode`

Ordre : id → token (via `ids_to_tokens`) → concaténation → caractère →
octet (`char_to_byte`) → `bytes` → `.decode("utf-8", errors="replace")`.

**Invariants**

- on ne décode **jamais** un token seul : 1 448 tokens du vocabulaire ne
  sont pas des séquences UTF-8 valides (`¡` = octet `0xA1`) ; un caractère
  peut être coupé entre deux tokens ;
- les tokens spéciaux ne doivent pas apparaître dans la sortie ;
- un id inconnu doit lever une erreur explicite (avec l'id), pas un `KeyError` ;
- un préfixe d'ids peut produire `�` : c'est normal, `errors="replace"`.

**Vérification** : `decode(encode(p)) == p` sur les 11 prompts (le texte
revient en NFC, ce qui est attendu).

---

## J5 — `encode`

Assemblage : normaliser, pré-tokeniser, puis pour chaque morceau : octets →
caractères → `_bpe` → `tokens_to_ids`.

**Invariants**

- aucun token spécial ajouté ;
- erreur explicite si un token fusionné est absent du vocabulaire ;
- les ids restent dans `0..151642` (les 26 added_tokens ne sont jamais
  produits) ;
- optionnel mais justifié dans le README : un cache local
  `morceau -> ids` (les mêmes mots reviennent dans les prompts).

---

## Branchement dans le programme

`src/core/llm.py` a deux points d'appel :

```
:28  ids = self.model.encode(text)[0].tolist()   ->  self.tokenizer.encode(text)
:57  return self.model.decode(token_ids)         ->  self.tokenizer.decode(token_ids)
```

`LargeLanguageModel` reçoit le tokenizer ; les ~10 appels `self.llm.encode`
du décodeur ne changent pas. `main()` construit le tokenizer avec
`get_path_to_vocab_file()` / `get_path_to_merges_file()` du modèle chargé
(c'est ce qui rend `--model` correct).

Critère de fin :

```bash
grep -rn "model\.encode\|model\.decode" src/     # doit être vide
```

---

## Déboguer un écart d'ids

Toujours dans le même ordre (du plus grossier au plus fin) :

1. **morceaux** : comparer tes morceaux à ceux de la référence sur la même
   chaîne (le tokenizer Python expose son pré-tokenizer). Écart ici → J2.
2. **un seul mot** : encoder un mot isolé de part et d'autre ; si ça
   diverge, c'est J3 (regarde les rangs des paires du mot).
3. **caractères spéciaux** : tester `"é"`, `"é"` en NFD, `"🙂"`, `"\t"`,
   la chaîne vide. Écart ici → J1 ou NFC.
4. **réduire** : trouver la plus courte sous-chaîne qui échoue encore.

| Symptôme | Cause |
|---|---|
| ids faux seulement avec des accents | NFC oubliée |
| `KeyError` en fin d'`encode` | token fusionné absent du vocabulaire |
| token long jamais produit | tu t'arrêtes sur « existe dans le vocab », pas sur « rang applicable » |
| `Ġ` présent dans la sortie de `decode` | tu renvoies la concaténation des tokens, sans repasser par les octets |
| `UnicodeDecodeError` | `errors="replace"` oublié |
| ids justes mais génération fausse | contexte assemblé différemment (`base` + `user` encodés séparément vs ensemble) |

## Fini quand

1. `PersonalTokenizer` s'instancie sans le SDK ;
2. 3 tests synthétiques verts (J2, J3, J4) ;
3. 5 tests d'intégration verts (11 prompts + 15 morceaux structurels) ;
4. `grep "model\.encode\|model\.decode" src/` vide ;
5. `tests/test_tokenizer.py` : 12 verts.
