*Bonjour, mon nom est Durian, et je fais de mon mieux pour vous aider à bien faire le Call Me Maybe.*

> Ce journal, c'est **mes mots, mes doutes et mes idees, a voix haut**. 
> Ici, je racconte tout ca que j'avait voulet savoir depuis le commence de ce projet.

# Chapitre Zero

Avec la documentation oficielle du Huggingface, il est possible de comprendre comment le Byte-Pair Encoding (BPE) fonctionne, ce qu'est tres important a l'heure de implementer tes propre tokenizer.

Le site est: https://huggingface.co/learn/llm-course/en/chapter6/5

Maintenant, je ne suis pas quelle est la difference entre generer une phrase (Verifier quelle est le logit con le score le plus haut on fonctionne des token_ids qu'on a recu, et le ajoute a la phrase que passarait a l'encode prochaine)

-Calculer la quantite maximale des characters que je doit utiliser
-Calculer la largesse maximale du token que je peux generer avec une bucle autour de get_logits_from_ids()
-Faire une bucle seulement pour le nom de la fonction
-Acepter ou rejeter une logit_max, si ca ne satisfait pas une nome de fonctionne reelle, ou, s'il y a des echecs apres avoir essaye de creer une JSON avec ces neuvelles characteres.


# Chapitre Une

Ca que j'ai apris aussi aujourd'hui est ce que on peux aussi utiliser les fonctionnes json de la proxieme facon:

```
json.loads() parses a JSON string (or bytes/bytearray) and returns the corresponding Python object (dict, list, str, int, float, bool, or None). 

Basic usage

import json

s = '{"name": "John", "age": 30, "city": "New York"}'
data = json.loads(s)

print(data)          # {'name': 'John', 'age': 30, 'city': 'New York'}
print(data['age'])   # 30
print(type(data))    # <class 'dict'>
```

Aussi, The json.dumps() function in Python returns a JSON-formatted string representing the serialized Python object.  It does not write to a file but instead keeps the data in memory as a string, which is useful for APIs, logging, or database storage. 

```
Basic Example

import json

data = {"name": "John", "age": 30, "city": "New York"}
json_string = json.dumps(data)

print(json_string)
# Output: {"name": "John", "age": 30, "city": "New York"}
```

# Chapitre Deux

Voici ca que les autres ont fait dans ces programmes, avec mes mots:

- Parser les args
- Cache les definitions des fonctionnes et les prompts
- Preparer le model
- Preparer la liste de output
- Pour cheque prompt, creer une FunctionCall
- Ajoute le FunctionCall sur le result
- Ecrir le result

# Chapitre Trois

J'ai essayer de faire un echose, alors une fois le prompt est pret pour commencer a trouver le nom du fonctionne, qu'est ce que ce passe?

On fait, ca n'est pas mieux, mais et plus directe demander le nome du fonctionne selon la premier lettre du nome

Le object Vocab est tres important, car il est un conteneteur pour les masks que on apliquerait sur les logits. 

Comment constroire une Vocab?

Le Vocab plus simple a seulement les ids et les text du cheque token, et a la inverse aussi: donc:

Vocab:

self.tokens_to_ids = json.loads(get_path_to_logits().content())
self.ids_to_tokens = { v: k for k, v in self.tokens_to_ids}

donc on peux chercher les characters que aparraint dans cheque item du dict.

search_chars

returns all the positions where token contains any of the characters.
returns positions where token contains all of characters.
returns positions where token is exactly this character.

I want to:

Put false where the token ids does not match the JSON_CHUNK_REGEXP:

that is, it is not a valid string value for JSON, it is, it is an ecaped sequence. So all the valid JSON strings are normal characters or full escaped sequences...

Qu'est que je deux faire aujourd'hui?
Ca serait tres cool de avoir le fly-in delivere, et pouvoir faire le progres avec Call Me Maybe, mais pour moi c'est plus attractiv faire le progress avec mon LLM projet, d'une facon que je peux aussi apprendre quelque chose sur le RAG, si possible.

Je ne parle pas meme resolvre le projet, mais APPRENDRE seulement une chose bien de ce projet, pour pouvoir le controir petit a petit.

On ce moment, l'objectif est d'obtenir une file JSON en sortie qui est ecrit d'une facon valide.

Alors, je connais bien deux manieres de faire cette generation du json, l'une avec une machine a etat (finit state machine), et l'autre par contre genere un objet FunctionCall directement, je prefere generer du texte json, puisque on doit faire ca toujours et permettre a ma function de faire une seule chose.

Donc, une autre fonction validera ce json et construira l'objet fonctino, se necessaire

Comment puis-je utiliser un machine a etat dans ma solution?

La machine doit faire une chose selon son etat actuel, elle commence par prendre le nom de la fonciton a creer:

FSM.State = FN_NAME
ajoute le base_context_ids, puis le nom de la fonction
se aucun eche n'est survenu, elle passe a l'etat suivant

FMS.State = FN_PARAMS
ajouter le nom du parametre actuel a data_str
combiner les previous_ids avec cette nouvelle data a dedus
trouver le value pour ce parametre, a l'aide des fonctions generate_string() ou generate_number()
selon son type, ajouter la valeur en tant que string a data_str
S'il n'y a plus de parametres, change l'etat en FMS.State = FINISH
Sinon, continue avec le parametre suivant, changer le parametre_index

Objectif: ajouter

# Chapitre Quattre: le vocab.py

Vocab.py est une morceaux tres important dans ce projet, je veux dire, c'est le propre Constraint Decode! car sa responsabilite est donnee les masks pour pouvoir rejete les tokens que n'interessant pas pour constroire une string, number, boolean ou integer 

Pense aussi, que le generation des tokens est tres trivial: On ne pide les tokens au model que dans les situations suivants:

1. on est en train de generer les "values" des champs "parameters" 

2. en train de adiniver le nome du fonction selectione

vocab.py — Le cœur du constrained decoding
Le principe
À chaque étape de génération, le modèle produit un vecteur de logits (un score par token du vocabulaire, ~151 643 valeurs). vocab.py fournit un masque binaire de la même taille :

logits:  [ 0.3, -1.2,  4.7,  0.1, ... ]   ← 151 643 valeurs
masque:  [  1,    0,    0,    1, ... ]   ← 1 = autorisé, 0 = interdit

On applique le masque avant le sampling :

logits = logits.masked_fill(mask == 0, float("-inf"))
# → les tokens interdits ont probabilité 0, impossible qu'ils soient tirés

Le modèle ne voit jamais un token qu'il ne devrait pas produire.

Pourquoi la génération est "triviale"
Tu n'appelles le modèle que dans deux situations :

┌─────────────────────────────────────────────────────────┐
│  Situation 1 : choix du nom de fonction                 │
│                                                         │
│  Le masque = { ids des tokens qui composent les noms    │
│                des fonctions valides }                  │
│  → le modèle ne peut produire QUE un nom existant       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Situation 2 : génération de la valeur d'un paramètre   │
│                                                         │
│  Le masque dépend du type déclaré dans la               │
│  définition de la fonction :                            │
│    "type": "string"  → masque string                    │
│    "type": "integer" → masque integer                   │
│    "type": "boolean" → masque boolean                   │
│                                                         │
│  → le modèle ne peut produire QUE une valeur du bon type│
└─────────────────────────────────────────────────────────┘

Tout le reste (les clés JSON, les virgules, les accolades, :, etc.) est déterministe — tu les écris directement, sans passer par le modèle. Le modèle ne "décide" que le contenu sémantique.


# Chapitre Sept: Creer te propre tokenisateur

Encode:

" Je suis une de les methods le plus important de ce projet"
tokens = self.split(" ")

Ca ne sirve pas, car pour faire une Tokenisateur, c'est deux choses.

Vraiment, pour le bonus on ne doit pas faire une Tokenizer propre, car c'est une procesus tres important pour une LLM, dans leuelle le model avraira son vocab.json et son merge.json

Le tokenizer a son propre trainement (avec une corpus) et algorithm de entrainement, appeller (BPE Byte Pair Encoding, Tokenization, WordPiece, etc), GPT-2 et Claude lorsque ca permettre representer ne importe quelle character affichable. Les autres models utilisant une UTF-8 byte fallback,  utilisant le BPE Tokenization, qui utilises les octes car ca permettre n'avoir pas une character <UNK>. GPT-2 et Claude, et autres models preferent utiliser petits tokens plutot que renvoyer le token UNKNOWN pluiseurs fois.


Donc, l'objectif du tokenizer et produir une file vocab.json, ou on trouver les tokens avec ses ids. Pour ca c'est important utiliser le merge.txt aussi.


Apres faire la normalisation e pre-tokenisation, quelle est le pas suivant?

Le premier pas est encoder les unicodes, car ca permettre former ne importe quelle sucession de characters avec les bytes initielles. Apres, ca, lorsque on a les ids initielles de notre vocabulaire, on commence pour passe l'algorithme de BPE a travers du corpus de trainement.


Le pas suivant est commencer a voir quelles fusions (merges) dans notre vocab sont elles dans le chunk que on est en train de verifier. Par exemple.

Apres avoir vu ces ids, on doit retorner ca et les ajouter a le array de sortie. 

# Chapitre Huit: Comment faire une visualisateur:

Ca que on voix faire por visualiser:

Afficher dans le ecran, et eraser tout: Ca donne la ilusion de movement.
On peut utiliser une callback pour motrer les choses dans le ecran.

Qu'est ce que je veux montrer?

Prompt Index
Prompt Count
Nombre de prompt
Tokens utilises
Phrase colle dans le buffer_str
Les tokens (ou chunks de string) qui sont ajoutes a le buffer, et d'ou y vien (si d'il model, ou de part du SCAFFOLD comment une string hardcode)