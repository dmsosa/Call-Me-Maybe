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