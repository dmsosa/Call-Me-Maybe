# Visualiseur TUI (`blessed`) — spec de construction

> Bonus du sujet (Chapitre VII) : *Visualization of the generation process*.
> Ce document est une **spec + fragments vérifiés**, pas une implémentation :
> le fichier `src/tui/gui.py` est à écrire par l'étudiant.

## 0. Dépendance

```bash
uv add blessed        # blessed==1.52.0 + jinxed + wcwidth
```

`blessed` = habillage de `curses` : des **chaînes magiques** (`term.move_xy(x, y)`)
que tu `print`, plus la gestion des touches et de l'écran alterné. Pas de widgets,
pas de layout : tout se positionne à la main.

## 1. Contrat de données

La vue ne lit **que** cet état. Tout ce qui n'y est pas ne peut pas être affiché.

```python
class Origin(StrEnum):
    SCAFFOLD = "scaffold"   # texte ecrit par nous  ({"prompt": , ", "parameters": {, })
    MODEL    = "model"      # token echantillonne sous contrainte
    INPUT    = "input"      # le prompt recopie

@dataclass
class Step:
    origin: Origin
    text: str               # ce morceau apporte au buffer
    ids: list[int]          # ce qui est reellement ajoute au contexte
    choices: int            # nb de tokens autorises au dernier pas (1 = force)
    top_raw: list[tuple[str, int, float]]    # top-5 logits AVANT masque
    top_masked: list[tuple[str, int, float]] # top-5 APRES masque
```

Ce que le décodeur doit exposer (aujourd'hui absent) :

| Besoin | État actuel | Action |
|---|---|---|
| `data_str` | variable locale `data`, jamais retournée (str immuable) | champ `data_str: str` alimenté par un `_add(origin, text)` |
| top-5 | seulement `masked` + `argmax` | calculer le top-5 là où `masked` existe |
| nb de choix | implicite dans le masque | `int(mask.sum())` |

Vérification qui vaut un test : à la fin d'un prompt,
`llm.decode(ids[base:]) == "".join(parts)`.

## 2. Géométrie (vérifiée à 60×18, 80×24, 100×26)

```
height : 0                titre (1)
         1                filet
         2..5             header : model / prompt i/n + cache + s/token / "> prompt"
         6                filet
         7..h-4           corps : [ buffer | top-5 ]   (panneaux cote a cote)
         h-3..h-1         footer : filet + boutons
```

```python
width  = min(term.width, 120)      # au-dela, les lignes sont illisibles
height = min(term.height, 40)
if width < 60 or height < 18:
    # message court + sortie : jamais de layout qui deborde
    ...
right_w = min(38, max(28, width // 3))   # panneau logits
split   = width - right_w - 1            # colonne du separateur
```

Toute ligne rendue doit faire **exactement** `width` caractères :
`assert len(row) == width` dans la fonction de rendu (c'est le garde-fou
qui attrape 90 % des bugs d'alignement).

## 3. Les trois couches

1. **Fond** : trame de « carrés » (un caractère de large, deux de haut
   d'aspect) → un point tous les 2 caractères. Terminaux : **aucune
   transparence réelle** ; « semi-transparent » = glyphe très peu contrasté
   (`·`, `░`) ou `term.color_rgb(80, 80, 80)`.
2. **Panneaux** : `fill()` (espaces) puis cadre `┌─┐│└┘`. Sans le `fill`,
   la trame passe sous le texte → illisible.
3. **Contenu** : texte + couleurs par morceau.

## 4. Mockup (sortie réelle du prototype)

```
                                           Call Me Maybe
────────────────────────────────────────────────────────────────────────────────────────────────────
·model  : Qwen/Qwen3-0.6B · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
·prompt : 7/11   cache: on   0.21 s/token · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
·> Somme de 12 et 30, puis multiplie le resultat par 3· · · · · · · · · · · · · · · · · · · · · · · ·
────────────────────────────────────────────────────────────────────────────────────────────────────
┌─ buffer (data_str) ─────────────────────────────────────────────┐┌─ top-5 logits ────────────────┐
│{"prompt": "Somme de 12 et 30, puis multiplie le resultat par 3",││␣a           64      11.24     │
│ "name": "fn_add_numbers", "parameters": {"a": 12                ││⏎            198      9.71     │
│                                                                 ││'b'          65       8.02     │
│                                                                 ││␣b           66       7.44     │
│                                                                 ││0            15       6.10     │
└─────────────────────────────────────────────────────────────────┘└───────────────────────────────┘
────────────────────────────────────────────────────────────────────────────────────────────────────
·[q] quit  [space] step  [a] auto  [o] output · · · · · · · · · · · · · · · · · · · · · · · · · · ·
```

## 5. Fragments vérifiés (hors terminal, purs)

```python
DOT = "\u00b7"   # '·'

def blank_grid(width: int, height: int, cw: int = 2) -> list[str]:
    row = "".join(DOT if x % cw == 0 else " " for x in range(width))
    return [row] * height

def put(rows: list[str], x: int, y: int, text: str) -> None:
    if not (0 <= y < len(rows)) or x >= len(rows[y]):
        return
    line = rows[y]
    text = text[: len(line) - max(0, x)]
    if x < 0:
        text, x = text[-x:], 0
    rows[y] = line[:x] + text + line[x + len(text):]

def fill(rows, x0, y0, x1, y1) -> None:          # panneau opaque
    for y in range(y0, y1 + 1):
        if 0 <= y < len(rows):
            put(rows, x0, y, " " * (x1 - x0 + 1))

def box(rows, x0, y0, x1, y1, title="") -> None:
    w = x1 - x0 + 1
    fill(rows, x0, y0, x1, y1)
    put(rows, x0, y0, "\u250c" + "\u2500" * (w - 2) + "\u2510")
    put(rows, x0, y1, "\u2514" + "\u2500" * (w - 2) + "\u2518")
    for y in range(y0 + 1, y1):
        put(rows, x0, y, "\u2502"); put(rows, x1, y, "\u2502")
    if title:
        put(rows, x0 + 2, y0, f" {title} "[: max(0, w - 4)])
```

Affichage des tokens byte-level (`Ġ`, `Ċ`, `ġ` sont **invisibles** à l'écran) :

```python
def sanitize_token(tok: str) -> str:
    return (tok.replace("\u0120", "\u2423")   # Ġ -> ␣
               .replace("\u010a", "\u23ce")   # Ċ -> ⏎
               .replace("\u0121", "?"))       # ġ -> ?
```

Top-5 sur les logits masqués (mesuré : `argpartition` **58 µs**,
`argsort` complet **268 µs** pour 151 936 logits ; forward pass 210 ms) :

```python
finite = np.flatnonzero(np.isfinite(masked))
if finite.size <= 5:
    top = finite
else:
    top = finite[np.argpartition(-masked[finite], 5)[:5]]
top = top[np.argsort(-masked[top])]
```

## 6. `CmmGui(BaseModel)` : ce que pydantic impose

```python
from pydantic import BaseModel, ConfigDict, Field
from blessed import Terminal

class CmmGui(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    term: Terminal = Field(default_factory=Terminal, exclude=True)  # sinon model_dump echoue
    model_name: str
    index: int = 0
    total: int = 0
    cache: str = "off"
    spt: float = 0.0
    prompt_text: str = ""
    data_str: str = ""
    top5: list[tuple[str, int, float]] = Field(default_factory=list)
```

- `arbitrary_types_allowed` : `Terminal` n'est pas un type pydantic connu.
- `exclude=True` : sinon `model_dump()` / `model_dump_json()` explose.
- pas de `__init__` maison : `model_post_init(self, _ctx)` (pydantic l'appelle).
- ne PAS activer `validate_assignment` : `data_str += …` à chaque token
  revaliderait tout (des milliers de fois).

Variante plus testable (recommandée) : `GuiState(BaseModel)` (données pures)
+ un renderer non-pydantic ; `render(state, width, height) -> list[str]` est
alors une **fonction pure**, testable sans terminal.

## 7. Boutons, touches, pas-à-pas

Un terminal n'a **pas** de clic. Un « bouton » = un libellé + un raccourci :

```
[q] quit   [space] step   [a] auto   [s] stats   [o] output
```

Rendu « bouton » : `term.reverse(" q ")` ou crochets `[ q ]`. La souris est
possible (`term.mouse_enabled()` + parsing `\x1b[<b;x;yM`) mais c'est du
travail pour rien ici.

Le blocage du pas-à-pas se fait **dans le callback**, pas dans le cœur :

```python
def on_step(self, step: Step) -> None:
    self.last = step
    self.draw()
    if self.step_mode:
        while True:
            key = term.inkey(timeout=None)    # bloquant : c'est voulu
            if key == " ":
                return
            if key == "a":
                self.step_mode = False
                return
            if key == "q":
                raise VisualizerQuit
```

Le cœur ne connaît ni `inkey` ni `blessed`, et il n'y a **aucun thread**
(dessiner coûte ~0,1 ms contre 210 ms de forward pass).

## 8. Pièges (dans l'ordre où ils mordent)

1. `term.move_xy(x, y)` = (colonne, ligne) — pas (ligne, colonne).
2. Pas de transparence : fond = glyphes peu contrastés, panneaux = `fill()`.
3. Largeur d'affichage ≠ `len()` : `Ġ`/`Ċ`, émojis (2), accents combinants (0).
   Utiliser `term.length(s)` ou sanitizer les tokens.
4. Coloriser **puis** `ljust` = padding faux (les codes ANSI comptent).
   Pad la ligne brute, colorise ensuite.
5. `fullscreen()` / `cbreak()` / `hidden_cursor()` dans un `with` : sinon le
   terminal reste cassé après une exception.
6. `if not sys.stdout.isatty(): return` dès le constructeur (`make moulinette`
   redirige stdout → sinon des séquences d'échappement dans les fichiers).
7. Le résultat final s'affiche **après** le `with` (l'écran alterné l'efface).
8. `term.width/height` relus à chaque frame (resize) → jamais de valeur globale.
9. Toute ligne doit faire `width` exactement (`assert` dans le rendu).
10. `-inf` dans les logits : filtrer `np.isfinite` avant `argpartition`.

## 9. Tests sans terminal

```python
def test_frame_has_exact_dimensions() -> None:
    rows = render(state, 80, 24)
    assert len(rows) == 24
    assert {len(r) for r in rows} == {80}

def test_frame_contains_the_required_fields() -> None:
    text = "\n".join(render(state, 100, 30))
    for needle in ("Call Me Maybe", "Qwen", "7/11", "cache", "s/token",
                   "top-5", "data_str"):
        assert needle in text
```

Aucun de ces tests n'a besoin de `blessed`, de TTY, ni du modèle.

## 10. Plan

| Jalon | Contenu | Critère |
|---|---|---|
| J1 | `render(state, w, h) -> list[str]` pure, fausses données | tests §9 verts, mockup fidèle |
| J2 | `draw()` : `blessed`, écran alterné, `q` quitte, resize | terminal intact après `q` |
| J3 | `_add`/`data_str`/`top5` dans le décodeur + couleur par origine | buffer qui pousse, top-5 vivant |
| J4 | pas-à-pas + boutons + `--visualize`/`--step` + garde `isatty` | démontrable en soutenance |

## 11. Flow des événements (`on_change`)

### Qui possède quoi

| Donnée affichée | Propriétaire | Comment la vue l'obtient |
|---|---|---|
| `model_name` | `main` / `llm` (`--model`) | injecté dans la vue au démarrage |
| `prompt_index`, `total` | `Generator.run` (`enumerate(self.prompts)`) | le Generator enrichit l'événement |
| `cache` (on/off, hit/miss) | `Generator` (`cache_results`, `state`) | idem |
| `spt` | `Generator` (chrono par prompt) | idem |
| texte du prompt | `Generator` (`prompt.prompt`) | idem |
| `data_str`, `parts`, `ids` | `ConstrainedDecoder` | événement `append` |
| `top5`, `choices`, `step` | `ConstrainedDecoder` | événement `append` |
| `state` (FN_PROMPT…) | `ConstrainedDecoder` | événement `state` |

Règle : **le décodeur émet ce qu'il sait, le Generator ajoute le contexte du
run.** Le décodeur ne doit jamais connaître les prompts, le cache ni le chrono.

### Les deux granularités

```
_generate_json
 └─ while state != FINISH
     ├─ _handle_state          <- 4 fois par prompt  => evenement "state"
     │   ├─ _handle_fn_name
     │   └─ _handle_params
     │       ├─ generate_number   <- boucle interne : 1 append par chiffre
     │       └─ generate_string   <- 1 append par token
     └─ _add(origin, text)     <- 1 fois par morceau => evenement "append"
```

Un hook placé dans `_handle_state` ne voit donc **pas** les tokens : il ne
verrait que les 4 transitions d'état. Le point d'émission correct est
`_add()`, le **goulot unique** par lequel passent tous les ajouts de texte/ids.

### Le flot complet

```
main --visualize
  └─ gui = CmmGui(...)                      # isatty() verifie ici
  └─ Generator(..., on_change=gui.update)   # callback = methode de la vue
      └─ run(): for i, prompt in enumerate(prompts)
          ├─ emit RunEvent(prompt_start, index, total, cache)
          └─ decoder.generate_json(prompt)
              └─ _add(origin, text):
                   1. met a jour data_str / parts / ids   (muter)
                   2. emit DecodeEvent(step, origin, text, ids, choices, top5)
                        -> Generator._forward(ev)   # ajoute index/total/cache
                            -> gui.update(frame)    # stocke, redessine
                                -> (mode pas-a-pas) inkey() bloquant
```

Émission **synchrone** : la vue tourne dans le thread de génération. C'est un
avantage (pause et pas-à-pas gratuits), à condition que `_add` ne fasse
**aucun I/O** : il appelle la callback, la vue décide quand dessiner.

### Muter PUIS émettre

```python
def _add(self, origin: Origin, text: str) -> None:
    ids = self.llm.encode(text)
    self.token_ids.extend(ids)          # 1. etat a jour
    self.parts.append(Part(origin, text, ids))
    self.data_str += text
    if self.on_change is not None:      # 2. puis notification
        self.on_change(self._event(origin, text, ids))
```

Inverser les deux lignes donne une vue qui affiche l'état *précédent* — le
symptôme classique : « il manque toujours le dernier token ».

### Test du sink (obligatoire)

```python
events: list[DecodeEvent] = []
decoder.on_change = events.append
assert decoder.generate_json(prompt)
assert [e.step for e in events] == list(range(len(events)))
assert "".join(e.text for e in events) == decoder.data_str
assert "".join(e.text for e in events) == llm.decode(ids[base:])
```

Si la dernière assertion casse : un morceau est ajouté sans passer par `_add()`.
