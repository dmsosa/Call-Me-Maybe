"""Tests du décodeur contraint, SANS charger le modèle.

Le décodeur ne dépend du modèle que par deux méthodes : ``encode`` et
``logits``. On peut donc lui donner un LLM factice et tester toute la
machine à états (échappement du prompt, choix du nom, nombres, chaînes)
en quelques millisecondes.

``ConstrainedDecoder.model_construct`` construit l'objet sans valider les
types pydantic : c'est ce qui permet de passer les faux objets.
"""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.constrained_decoder import ConstrainedDecoder
from src.model.schema import FnDefinition, ParamSpec

N_LOGITS = 200


class FakeLLM:
    """LLM minimal : encode/décode caractère par caractère, logits scriptés.

    ``logits_script`` est une liste de dict {id_token: score} consommée un
    élément par appel. Un appel non prévu lève une erreur : le test échoue
    si le décodeur interroge le modèle là où il ne devrait pas.
    """

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_char: dict[int, str] = {}
        self.logits_script: list[dict[int, float]] = []
        self.calls = 0

    def _id(self, char: str) -> int:
        if char not in self._to_id:
            new_id = len(self._to_id) + 10
            self._to_id[char] = new_id
            self._to_char[new_id] = char
        return self._to_id[char]

    def encode(self, text: str) -> list[int]:
        return [self._id(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._to_char[token_id] for token_id in ids)

    def logits(self, ids: list[int]) -> np.ndarray:
        assert self.logits_script, "le modèle ne devait pas être appelé ici"
        self.calls += 1
        step = self.logits_script.pop(0)
        out = np.zeros(N_LOGITS, dtype=np.float32)
        for token_id, score in step.items():
            out[token_id] = score
        return out


class FakeVocab:
    """Vocabulaire minuscule : mêmes attributs que ``Vocab``, sans fichier."""

    def __init__(self) -> None:
        self.n_logits = N_LOGITS
        self.ids_to_tokens: dict[int, str] = {}
        self.tokens_to_ids: dict[str, int] = {}
        self.digit_mask = np.zeros(N_LOGITS, dtype=bool)
        self.minus_mask = np.zeros(N_LOGITS, dtype=bool)
        self.dot_mask = np.zeros(N_LOGITS, dtype=bool)
        self.end_number_mask = np.zeros(N_LOGITS, dtype=bool)
        self.str_valid = [np.zeros(N_LOGITS, dtype=bool) for _ in range(2)]
        self.str_closed = [
            np.full(N_LOGITS, -1, dtype=np.int32) for _ in range(2)
        ]
        self.str_escaped = [np.zeros(N_LOGITS, dtype=bool) for _ in range(2)]
        # ids "spéciaux" que le décodeur utilise directement.
        self.quote_id = 30
        self.true_id = 40
        self.false_id = 41


def build_decoder(
    fn_by_name: dict[str, FnDefinition] | None = None,
    llm: FakeLLM | None = None,
    vocab: FakeVocab | None = None,
) -> ConstrainedDecoder:
    """Construit un décodeur avec des dépendances factices."""
    return ConstrainedDecoder.model_construct(
        llm=llm if llm is not None else FakeLLM(),
        vocab=vocab if vocab is not None else FakeVocab(),
        fn_by_name=fn_by_name if fn_by_name is not None else {},
    )


def one_noop_function() -> dict[str, FnDefinition]:
    """Une seule fonction, sans paramètre : aucun appel au modèle."""
    fn = FnDefinition(name="fn_ping", description="", parameters={})
    return {"fn_ping": fn}


# --------------------------------------------------------------------------
# Bug n°2 : le champ "prompt" doit être une chaîne JSON valide.
# --------------------------------------------------------------------------


def prompt_value(raw: str) -> object:
    """Extrait UNIQUEMENT la valeur du champ ``"prompt"`` du texte produit.

    On décode une valeur JSON à partir de la position qui suit
    ``"prompt": ``, sans exiger que tout le document soit valide : ce test
    reste ainsi focalisé sur l'échappement (les autres défauts du document
    ont leurs propres tests, un défaut = un test rouge).
    """
    marker = '"prompt": '
    start = raw.index(marker) + len(marker)
    value, _end = json.JSONDecoder().raw_decode(raw, start)
    return value


def test_prompt_with_double_quotes_is_escaped() -> None:
    decoder = build_decoder(one_noop_function())
    prompt = 'Replace all numbers in "Hello 34" with NUMBERS'

    assert prompt_value(decoder.generate_json(prompt)) == prompt


def test_prompt_with_backslash_keeps_the_backslash() -> None:
    decoder = build_decoder(one_noop_function())
    prompt = r"open C:\temp\new today"

    assert prompt_value(decoder.generate_json(prompt)) == prompt


def test_prompt_with_newline_is_escaped() -> None:
    decoder = build_decoder(one_noop_function())
    prompt = "first line\nsecond line"

    assert prompt_value(decoder.generate_json(prompt)) == prompt


def test_prompt_with_control_characters_is_escaped() -> None:
    decoder = build_decoder(one_noop_function())
    prompt = "tab\there and bell\x07"

    assert prompt_value(decoder.generate_json(prompt)) == prompt


def test_document_is_complete_json_for_a_zero_parameter_function() -> None:
    """Une fonction sans paramètre doit fermer ``parameters`` ET la racine."""
    decoder = build_decoder(one_noop_function())

    raw = decoder.generate_json("Ping")

    assert json.loads(raw) == {
        "prompt": "Ping",
        "name": "fn_ping",
        "parameters": {},
    }


# --------------------------------------------------------------------------
# Choix du nom : élimination par tokens, sans modèle quand un seul candidat.
# --------------------------------------------------------------------------


def test_single_function_needs_no_model_call() -> None:
    llm = FakeLLM()
    decoder = build_decoder(one_noop_function(), llm=llm)

    assert decoder.select_best_fn_name([]) == "fn_ping"
    assert llm.calls == 0


def test_name_is_chosen_among_the_masked_candidates() -> None:
    functions = {
        "fn_add_numbers": FnDefinition(name="fn_add_numbers"),
        "fn_greet": FnDefinition(name="fn_greet"),
    }
    llm = FakeLLM()
    decoder = build_decoder(functions, llm=llm)
    # Les deux noms divergent dès le premier token : 100 vs 150.
    decoder.fn_name_ids = {
        "fn_add_numbers": [100, 101],
        "fn_greet": [150, 151],
    }
    llm.logits_script = [{150: 9.0}]  # le modèle préfère fn_greet

    assert decoder.select_best_fn_name([7, 8]) == "fn_greet"
    assert llm.calls == 1


# --------------------------------------------------------------------------
# Générateurs de valeurs : masque appliqué, token d'arrêt respecté.
# --------------------------------------------------------------------------


def test_generate_number_stops_on_the_end_token() -> None:
    vocab = FakeVocab()
    vocab.ids_to_tokens = {
        10: "4",
        11: "2",
        12: "-",
        13: ".",
        14: ",",
        15: "}",
    }
    vocab.digit_mask[[10, 11]] = True
    vocab.minus_mask[12] = True
    vocab.dot_mask[13] = True
    vocab.end_number_mask[[14, 15]] = True

    llm = FakeLLM()
    llm.logits_script = [{10: 9.0}, {14: 9.0}]  # "4" puis fin de nombre

    decoder = build_decoder(one_noop_function(), llm=llm, vocab=vocab)

    assert decoder.generate_number([]) == "4"
    assert llm.calls == 2


def test_generate_string_closes_the_quote() -> None:
    vocab = FakeVocab()
    vocab.ids_to_tokens = {20: "ok", 21: '",'}
    vocab.str_valid[0][[20, 21]] = True
    vocab.str_closed[0][21] = 0  # la quote est le 1er caractère du token

    llm = FakeLLM()
    llm.logits_script = [{20: 9.0}, {21: 9.0}]

    decoder = build_decoder(one_noop_function(), llm=llm, vocab=vocab)

    value = decoder.generate_string([])

    assert value == '"ok"'
    assert json.loads(value) == "ok"


# --------------------------------------------------------------------------
# Assemblage complet : le JSON produit doit respecter les types déclarés.
# --------------------------------------------------------------------------


def number_vocab() -> FakeVocab:
    """Vocabulaire où "2.5" est un nombre et "," termine un nombre."""
    vocab = FakeVocab()
    vocab.ids_to_tokens = {10: "2", 11: "5", 12: ".", 14: ",", 15: "}"}
    vocab.digit_mask[[10, 11]] = True
    vocab.dot_mask[12] = True
    vocab.end_number_mask[[14, 15]] = True
    return vocab


def string_vocab() -> FakeVocab:
    """Vocabulaire où le token 21 contient la quote fermante."""
    vocab = FakeVocab()
    vocab.ids_to_tokens = {20: "shrek", 21: '",'}
    vocab.str_valid[0][[20, 21]] = True
    vocab.str_closed[0][21] = 0
    vocab.tokens_to_ids['"'] = 30
    return vocab


def test_string_parameter_is_not_double_quoted() -> None:
    functions = {
        "fn_greet": FnDefinition(
            name="fn_greet", parameters={"name": ParamSpec(type="string")}
        )
    }
    llm = FakeLLM()
    llm.logits_script = [{20: 9.0}, {21: 9.0}]
    decoder = build_decoder(functions, llm=llm, vocab=string_vocab())

    data = json.loads(decoder.generate_json("Greet shrek"))

    assert data["parameters"] == {"name": "shrek"}


def test_number_parameter_is_a_json_number() -> None:
    functions = {
        "fn_add_numbers": FnDefinition(
            name="fn_add_numbers", parameters={"a": ParamSpec(type="number")}
        )
    }
    llm = FakeLLM()
    llm.logits_script = [{10: 9.0}, {12: 9.0}, {11: 9.0}, {14: 9.0}]
    decoder = build_decoder(functions, llm=llm, vocab=number_vocab())

    data = json.loads(decoder.generate_json("What is 2.5?"))

    assert data["parameters"] == {"a": 2.5}
    assert isinstance(data["parameters"]["a"], float)


def test_two_number_parameters_are_separated_by_a_comma() -> None:
    functions = {
        "fn_add_numbers": FnDefinition(
            name="fn_add_numbers",
            parameters={
                "a": ParamSpec(type="number"),
                "b": ParamSpec(type="number"),
            },
        )
    }
    llm = FakeLLM()
    llm.logits_script = [
        {10: 9.0},
        {12: 9.0},
        {11: 9.0},
        {14: 9.0},  # a = 2.5
        {10: 9.0},
        {14: 9.0},  # b = 2
    ]
    decoder = build_decoder(functions, llm=llm, vocab=number_vocab())

    data = json.loads(decoder.generate_json("sum"))

    assert data["parameters"] == {"a": 2.5, "b": 2.0}


def test_boolean_parameter_is_supported() -> None:
    functions = {
        "fn_flag": FnDefinition(
            name="fn_flag", parameters={"on": ParamSpec(type="boolean")}
        )
    }
    llm = FakeLLM()
    llm.logits_script = [{40: 9.0}]  # true_id
    decoder = build_decoder(functions, llm=llm)

    data = json.loads(decoder.generate_json("flag"))

    assert data["parameters"] == {"on": True}


def test_two_prompts_in_a_row_both_work() -> None:
    decoder = build_decoder(one_noop_function())

    first = decoder.generate_json("one")
    second = decoder.generate_json("two")

    assert (prompt_value(first), prompt_value(second)) == ("one", "two")


def test_system_prompt_lists_every_function_and_its_parameters() -> None:
    llm = FakeLLM()
    functions = {
        "fn_greet": FnDefinition(
            name="fn_greet",
            description="Say hi",
            parameters={"name": ParamSpec(type="string")},
        ),
        "fn_add_numbers": FnDefinition(
            name="fn_add_numbers",
            parameters={
                "a": ParamSpec(type="number"),
                "b": ParamSpec(type="number"),
            },
        ),
    }
    decoder = build_decoder(functions, llm=llm)

    context = decoder.base_context_str

    assert "fn_greet(name: string) - Say hi" in context
    assert "fn_add_numbers(a: number, b: number)" in context


def test_fn_name_ids_are_precomputed_for_every_function() -> None:
    llm = FakeLLM()
    functions = {
        "fn_greet": FnDefinition(name="fn_greet"),
        "fn_add_numbers": FnDefinition(name="fn_add_numbers"),
    }
    decoder = build_decoder(functions, llm=llm)

    assert decoder.fn_name_ids == {
        "fn_greet": llm.encode("fn_greet"),
        "fn_add_numbers": llm.encode("fn_add_numbers"),
    }


# --------------------------------------------------------------------------
# Cas limites des masques : entier, signe, séquence d'échappement.
# --------------------------------------------------------------------------


def test_integer_parameter_never_contains_a_dot() -> None:
    llm = FakeLLM()
    # Après "2", le modèle préfère "." : interdit pour un entier.
    llm.logits_script = [{10: 9.0}, {12: 9.0}, {10: 9.0}]
    decoder = build_decoder(one_noop_function(), llm=llm, vocab=number_vocab())
    decoder.token_limits = {**decoder.token_limits, "integer": 3}

    value = decoder.generate_number([], integer=True)

    assert "." not in value
    assert value.isdigit()


def test_minus_sign_is_only_allowed_as_first_character() -> None:
    vocab = number_vocab()
    vocab.ids_to_tokens[13] = "-"
    vocab.minus_mask[13] = True
    llm = FakeLLM()
    # "-", puis "2", puis fin de nombre (le "-" final est impossible).
    llm.logits_script = [{13: 9.0}, {10: 9.0}, {14: 9.0}]
    decoder = build_decoder(one_noop_function(), llm=llm, vocab=vocab)

    assert decoder.generate_number([]) == "-2"


def test_string_escape_sequence_is_completed_by_the_next_token() -> None:
    vocab = FakeVocab()
    vocab.ids_to_tokens = {24: "a\\", 23: "n", 21: '",'}
    vocab.str_valid[0][[24, 23, 21]] = True
    vocab.str_escaped[0][24] = 1  # "a\" laisse une séquence ouverte
    vocab.str_valid[1][[23, 21]] = True  # état "escape" : échappements
    vocab.str_closed[0][21] = 0
    vocab.str_closed[1][21] = 0
    llm = FakeLLM()
    llm.logits_script = [{24: 9.0}, {23: 9.0}, {21: 9.0}]
    decoder = build_decoder(one_noop_function(), llm=llm, vocab=vocab)

    value = decoder.generate_string([])

    assert value == '"a\\n"'
    assert json.loads(value) == "a\n"


def test_select_best_token_raises_when_no_token_is_allowed() -> None:
    decoder = build_decoder(one_noop_function())

    with pytest.raises(ValueError):
        decoder.select_best_token(np.full(N_LOGITS, -np.inf))


def test_unterminated_string_is_never_returned_as_valid_json() -> None:
    """Si le modèle n'écrit pas la quote fermante, le JSON reste valide."""
    vocab = FakeVocab()
    vocab.ids_to_tokens = {20: "ab"}
    vocab.str_valid[0][20] = True
    llm = FakeLLM()
    llm.logits_script = [{20: 9.0}]
    functions = {
        "fn_greet": FnDefinition(
            name="fn_greet", parameters={"name": ParamSpec(type="string")}
        )
    }
    decoder = build_decoder(functions, llm=llm, vocab=vocab)
    decoder.token_limits = {**decoder.token_limits, "string": 1}

    try:
        raw = decoder.generate_json("p")
    except ValueError:
        return  # refuser clairement est acceptable

    json.loads(raw)  # sinon, le document doit être un JSON valide


def test_integer_parameter_end_to_end() -> None:
    functions = {
        "fn_add": FnDefinition(
            name="fn_add", parameters={"a": ParamSpec(type="integer")}
        )
    }
    llm = FakeLLM()
    llm.logits_script = [{10: 9.0}, {14: 9.0}]
    decoder = build_decoder(functions, llm=llm, vocab=number_vocab())

    data = json.loads(decoder.generate_json("p"))

    assert data["parameters"] == {"a": 2}
    assert isinstance(data["parameters"]["a"], int)


def test_number_cannot_start_with_a_dot() -> None:
    llm = FakeLLM()
    llm.logits_script = [{12: 9.0}, {14: 9.0}]
    decoder = build_decoder(one_noop_function(), llm=llm, vocab=number_vocab())
    decoder.token_limits = {**decoder.token_limits, "number": 2}

    value = decoder.generate_number([])

    assert "." not in value


def test_empty_string_value_stays_valid_json() -> None:
    vocab = FakeVocab()
    vocab.ids_to_tokens = {21: '",'}
    vocab.str_valid[0][21] = True
    vocab.str_closed[0][21] = 0
    llm = FakeLLM()
    llm.logits_script = [{21: 9.0}]
    functions = {
        "fn_greet": FnDefinition(
            name="fn_greet", parameters={"name": ParamSpec(type="string")}
        )
    }
    decoder = build_decoder(functions, llm=llm, vocab=vocab)

    data = json.loads(decoder.generate_json("p"))

    assert data["parameters"] == {"name": ""}


def test_parameter_name_with_a_quote_stays_valid_json() -> None:
    """Le nom d'un paramètre vient du fichier : il est validé au schéma.

    Voir ``test_schema.py`` : ``FnDefinition`` refuse un nom qui ne
    respecte pas le snake_case, donc le décodeur ne peut jamais
    écrire un nom brut dans le JSON.
    """
    with pytest.raises(ValidationError):
        FnDefinition(
            name="fn_odd", parameters={'a"b': ParamSpec(type="number")}
        )


def test_trailing_dot_at_the_token_limit_is_removed() -> None:
    """"2." n'est pas un nombre : le point final doit disparaître."""
    llm = FakeLLM()
    llm.logits_script = [{10: 9.0}, {12: 9.0}]
    decoder = build_decoder(one_noop_function(), llm=llm, vocab=number_vocab())
    decoder.token_limits = {**decoder.token_limits, "number": 2}

    assert decoder.generate_number([]) == "2"


# --------------------------------------------------------------------------
# Noms de fonction : cas où un nom est le préfixe d'un autre.
# --------------------------------------------------------------------------


def test_shorter_name_wins_when_the_model_stops_it() -> None:
    functions = {
        "fn_a": FnDefinition(name="fn_a"),
        "fn_ab": FnDefinition(name="fn_ab"),
    }
    llm = FakeLLM()
    decoder = build_decoder(functions, llm=llm)
    decoder.fn_name_ids = {"fn_a": [100], "fn_ab": [100, 101]}
    # fn_a s'arrête avec une quote (quote_id == 30), fn_ab continue.
    llm.logits_script = [{30: 9.0}]

    assert decoder.select_best_fn_name([]) == "fn_a"


def test_longer_name_wins_when_the_model_continues() -> None:
    functions = {
        "fn_a": FnDefinition(name="fn_a"),
        "fn_ab": FnDefinition(name="fn_ab"),
    }
    llm = FakeLLM()
    decoder = build_decoder(functions, llm=llm)
    decoder.fn_name_ids = {"fn_a": [100], "fn_ab": [100, 101]}
    llm.logits_script = [{101: 9.0}]

    assert decoder.select_best_fn_name([]) == "fn_ab"
