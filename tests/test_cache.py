"""Contrat du cache L1 : (prompt + définitions) -> FunctionCall, sur disque.

Le cache est indépendant du modèle : on peut tester la clé, la lecture, la
revalidation et la persistance sans charger quoi que ce soit.
"""
import json
from pathlib import Path
from typing import Any

from src.core.cache import CACHE_VERSION, CmmCache
from src.core.generator import Generator
from src.model.metrics import CmmMetrics
from src.model.schema import (
    FnDefinition,
    FunctionCall,
    ParamSpec,
    PromptDefinition,
)
from src.storage.json_writer import JsonWriter

FNS = [
    FnDefinition(
        name="fn_add",
        parameters={
            "a": ParamSpec(type="number"),
            "b": ParamSpec(type="number"),
        },
    ),
    FnDefinition(
        name="fn_greet", parameters={"name": ParamSpec(type="string")}
    ),
]
BY_NAME = {fn.name: fn for fn in FNS}

SUM_JSON = (
    '{"prompt": "2+3", "name": "fn_add", '
    '"parameters": {"a": 2, "b": 3}}'
)


def cache_at(tmp_path: Path, **kwargs: Any) -> CmmCache:
    return CmmCache(path=tmp_path / "answers.json", enabled=True, **kwargs)


def sum_call(prompt: str = "2+3") -> FunctionCall:
    return FunctionCall(
        prompt=prompt, name="fn_add", parameters={"a": 2.0, "b": 3.0}
    )


# ---------------------------------------------------------------- la clé


def test_key_is_stable_across_instances(tmp_path: Path) -> None:
    first = cache_at(tmp_path)
    second = cache_at(tmp_path)

    assert first.cache_key("2+3", FNS) == second.cache_key("2+3", FNS)


def test_key_changes_with_the_prompt(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)

    assert cache.cache_key("2+3", FNS) != cache.cache_key("2+4", FNS)


def test_key_changes_when_a_parameter_type_changes(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    renamed = [
        FnDefinition(
            name="fn_add",
            parameters={
                "a": ParamSpec(type="number"),
                "b": ParamSpec(type="integer"),
            },
        )
    ]

    assert cache.cache_key("2+3", FNS) != cache.cache_key("2+3", renamed)


def test_key_changes_when_the_version_changes(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    future = CmmCache(
        path=tmp_path / "answers.json",
        enabled=True,
        version=CACHE_VERSION + 1,
    )

    assert cache.cache_key("2+3", FNS) != future.cache_key("2+3", FNS)


# ------------------------------------------------------------- le disque


def test_missing_file_means_an_empty_cache(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)

    assert cache.entries == {}
    assert cache.hits == 0


def test_store_persists_a_reloadable_entry(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    cache.store("2+3", FNS, sum_call())

    raw = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    reloaded = cache_at(tmp_path)

    assert raw["version"] == CACHE_VERSION
    assert len(raw["entries"]) == 1
    assert len(reloaded.entries) == 1


def test_corrupt_json_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text("{ pas du json", encoding="utf-8")

    cache = cache_at(tmp_path)

    assert cache.entries == {}


def test_version_mismatch_ignores_all_entries(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    entry = {"prompt": "2+3", "name": "fn_add", "parameters": {"a": 2.0}}
    path.write_text(
        json.dumps({"version": CACHE_VERSION + 99, "entries": {"k": entry}}),
        encoding="utf-8",
    )

    cache = cache_at(tmp_path)

    assert cache.entries == {}, (
        "un cache d'une autre version ne doit jamais être chargé"
    )


def test_invalid_entry_is_dropped_and_valid_one_kept(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    good = {"prompt": "2+3", "name": "fn_add", "parameters": {"a": 2.0}}
    path.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "entries": {"good": good, "bad": {"prompt": "x"}},
            }
        ),
        encoding="utf-8",
    )

    cache = cache_at(tmp_path)

    assert list(cache.entries) == ["good"]


def test_disabled_cache_never_writes(tmp_path: Path) -> None:
    cache = CmmCache(path=tmp_path / "answers.json", enabled=False)

    cache.store("2+3", FNS, sum_call())

    assert not (tmp_path / "answers.json").exists()


# -------------------------------------------------------------- lookup


def test_lookup_returns_the_stored_call(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    cache.store("2+3", FNS, sum_call())

    found = cache.lookup("2+3", FNS, BY_NAME)

    assert found is not None
    assert found.name == "fn_add"
    assert found.parameters == {"a": 2.0, "b": 3.0}
    assert found.prompt == "2+3"


def test_lookup_ignores_another_prompt(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    cache.store("2+3", FNS, sum_call())

    assert cache.lookup("4+5", FNS, BY_NAME) is None


def test_lookup_ignores_an_unknown_function(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    cache.store("2+3", FNS, sum_call())

    assert cache.lookup("2+3", FNS, {"fn_greet": FNS[1]}) is None


def test_lookup_ignores_a_wrong_parameter_type(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    entry = {"prompt": "2+3", "name": "fn_add",
             "parameters": {"a": "deux", "b": 3.0}}
    path.write_text(
        json.dumps({"version": CACHE_VERSION, "entries": {"k": entry}}),
        encoding="utf-8",
    )

    assert cache_at(tmp_path).lookup("2+3", FNS, BY_NAME) is None


def test_lookup_ignores_an_undeclared_parameter(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    call = FunctionCall(
        prompt="2+3", name="fn_add",
        parameters={"a": 2.0, "b": 3.0, "zzz": 1.0},
    )
    cache.store("2+3", FNS, call)

    assert cache.lookup("2+3", FNS, BY_NAME) is None


def test_hits_counts_only_successful_lookups(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    assert cache.lookup("2+3", FNS, BY_NAME) is None
    cache.store("2+3", FNS, sum_call())

    assert cache.lookup("2+3", FNS, BY_NAME) is not None
    assert cache.lookup("2+3", FNS, BY_NAME) is not None
    assert cache.lookup("autre", FNS, BY_NAME) is None
    assert cache.hits == 2


def test_clear_empties_the_file(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    cache.store("2+3", FNS, sum_call())

    cache.clear()

    raw = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    assert raw["entries"] == {}
    assert cache.lookup("2+3", FNS, BY_NAME) is None


# --------------------------------------------------------- intégration


class FakeCalls:
    """Compteur d'appels modèle, comme ``LargeLanguageModel.calls``."""

    def __init__(self) -> None:
        self.calls = 0


class ScriptedDecoder:
    """Décodeur scripté : une réponse par prompt, et on note les appels."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.seen: list[str] = []

    def generate_json(self, prompt: str) -> str:
        self.seen.append(prompt)
        if prompt not in self.answers:
            raise AssertionError(f"le modèle a été appelé pour {prompt!r}")
        return self.answers[prompt]


class CachedGenerator(Generator):
    """Generator hors-ligne qui construit un vrai cache (sans modèle)."""

    def model_post_init(self, context: Any) -> None:
        self._store_fn_names()
        self.prompt_count = len(self.prompts)
        self.cache = CmmCache(
            path=Path(self.cache_path), enabled=self.cache_enabled
        )


def build_cached(
    decoder: ScriptedDecoder,
    prompts: list[str],
    tmp_path: Path,
) -> CachedGenerator:
    """metrics_path doit être un Path : model_construct ne coerce pas."""
    return CachedGenerator.model_construct(
        llm=FakeCalls(),
        fn_list=FNS,
        prompts=[PromptDefinition(prompt=p) for p in prompts],
        writer=JsonWriter(output_path=tmp_path / "results.json"),
        decoder=decoder,
        cache_enabled=True,
        cache_path=str(tmp_path / "answers.json"),
        metrics=CmmMetrics.model_construct(
            metrics_path=tmp_path / "metrics.json"
        ),
    )


def test_second_run_makes_no_model_call(tmp_path: Path) -> None:
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    second.run()

    assert second.results[0].name == "fn_add"
    assert second.results[0].parameters == {"a": 2.0, "b": 3.0}
    assert second.metrics.prompts[0].cache == "hit"
    assert second.metrics.prompts[0].calls == 0


def test_fallback_is_never_cached(tmp_path: Path) -> None:
    failing = build_cached(ScriptedDecoder({"boom": ""}), ["boom"], tmp_path)
    failing.run()

    written = CmmCache(path=tmp_path / "answers.json", enabled=True)

    assert failing.errors_count == 1
    assert written.entries == {}, "un fallback ne doit jamais entrer en cache"


def test_run_stores_under_the_key_it_will_look_up(tmp_path: Path) -> None:
    """Le magasin et le lookup doivent calculer la même clé.

    ``run`` doit passer la *chaîne* du prompt à ``store`` : passer
    l'objet ``PromptDefinition`` produit une clé différente, donc un
    cache qui ne touche jamais.
    """
    generator = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    generator.run()

    raw = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    expected = cache_at(tmp_path).cache_key("2+3", FNS)

    assert list(raw["entries"]) == [expected]


# ------------------------------------------------- coherence cache / evenements


def test_store_uses_the_lookup_prompt_not_the_call_prompt(tmp_path: Path) -> None:
    """Une reponse dont le champ 'prompt' differe ne doit pas empoisonner la cle."""
    cache = cache_at(tmp_path)
    other = FunctionCall(
        prompt="un autre prompt", name="fn_add", parameters={"a": 1.0, "b": 2.0}
    )

    cache.store("2+3", FNS, other)

    assert cache.lookup("2+3", FNS, BY_NAME) is not None


def test_lookup_rejects_an_entry_missing_required_parameters(
        tmp_path: Path) -> None:
    """Un cache valide ne doit pas ressusciter un appel incomplet."""
    cache = cache_at(tmp_path)
    cache.store(
        "2+3", FNS, FunctionCall(prompt="2+3", name="fn_add", parameters={})
    )

    assert cache.lookup("2+3", FNS, BY_NAME) is None


def test_cache_hit_sends_prompt_start_before_the_result(tmp_path: Path) -> None:
    """Sinon la vue garde les parts du prompt precedent et son header perime."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    seen: list[Any] = []
    second.on_change = seen.append
    second.run()

    assert seen, "un hit doit quand meme produire un evenement"
    assert seen[0].event_type == "prompt_start"
    assert seen[0].cache == "hit"


def test_exactly_one_prompt_end_per_prompt_including_hits(tmp_path: Path) -> None:
    """Le compteur de progression de la vue depend de ce nombre."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    seen: list[Any] = []
    second.on_change = seen.append
    second.run()

    assert [c.event_type for c in seen].count("prompt_end") == 1


def test_store_records_the_prompt_of_the_key(tmp_path: Path) -> None:
    """L'entree doit memoriser le prompt de la CLE, pas celui du FunctionCall.

    Sinon ``lookup`` compare deux prompts differents et rate sa propre
    entree : le cache ne touche jamais.
    """
    cache = cache_at(tmp_path)
    other = FunctionCall(
        prompt="un autre prompt",
        name="fn_add",
        parameters={"a": 1.0, "b": 2.0},
    )

    cache.store("2+3", FNS, other)

    entry = cache.entries[cache.cache_key("2+3", FNS)]
    assert entry.prompt == "2+3"


def test_mismatched_call_prompt_still_hits_after_a_reload(tmp_path: Path) -> None:
    """Le defaut precedent doit aussi etre corrige sur le disque."""
    cache = cache_at(tmp_path)
    other = FunctionCall(
        prompt="un autre prompt",
        name="fn_add",
        parameters={"a": 1.0, "b": 2.0},
    )
    cache.store("2+3", FNS, other)

    reloaded = cache_at(tmp_path)

    assert reloaded.lookup("2+3", FNS, BY_NAME) is not None


def test_lookup_rejects_an_entry_missing_one_parameter(tmp_path: Path) -> None:
    """Avec un seul parametre sur deux, l'appel est incomplet."""
    cache = cache_at(tmp_path)
    cache.store(
        "2+3", FNS,
        FunctionCall(prompt="2+3", name="fn_add", parameters={"a": 2.0}),
    )

    assert cache.lookup("2+3", FNS, BY_NAME) is None


def test_lookup_accepts_a_function_without_parameters(tmp_path: Path) -> None:
    """La verification des parametres ne doit pas rejeter zero parametre."""
    noop = FnDefinition(name="fn_ping", parameters={})
    cache = cache_at(tmp_path)
    cache.store(
        "ping", [noop],
        FunctionCall(prompt="ping", name="fn_ping", parameters={}),
    )

    assert cache.lookup("ping", [noop], {"fn_ping": noop}) is not None


def test_prompt_start_on_a_miss_announces_the_miss(tmp_path: Path) -> None:
    """Meme defaut que le hit, vu de l'autre cote : l'evenement dit 'off'."""
    generator = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    seen: list[Any] = []
    generator.on_change = seen.append

    generator.run()

    assert seen[0].event_type == "prompt_start"
    assert seen[0].cache == "miss"


def test_prompt_start_on_a_hit_carries_prompt_and_index(tmp_path: Path) -> None:
    """Le header de la vue a besoin du prompt, de l'index et du total."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    seen: list[Any] = []
    second.on_change = seen.append
    second.run()

    start = seen[0]
    assert (start.prompt, start.prompt_index, start.total_prompts) == (
        "2+3", 1, 1,
    )


def test_a_hit_does_not_rewrite_the_cache(tmp_path: Path) -> None:
    """Un hit ne doit ni reecrire ni dupliquer l'entree."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()
    before = (tmp_path / "answers.json").read_text(encoding="utf-8")

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    second.run()

    after = (tmp_path / "answers.json").read_text(encoding="utf-8")
    assert after == before
    assert len(cache_at(tmp_path).entries) == 1


def test_prompt_start_reports_each_prompts_own_cache_state(
        tmp_path: Path) -> None:
    """Le 2e prompt ne doit pas annoncer l'etat du 1er.

    Cache : "2+3" est stocke, "4+5" ne l'est pas. Les deux prompt_start
    doivent dire "hit" puis "miss" (et non "off" puis "hit").
    """
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    other_json = (
        '{"prompt": "4+5", "name": "fn_add", '
        '"parameters": {"a": 4, "b": 5}}'
    )
    second = build_cached(
        ScriptedDecoder({"4+5": other_json}), ["2+3", "4+5"], tmp_path
    )
    seen: list[Any] = []
    second.on_change = seen.append

    second.run()

    starts = [c for c in seen if c.event_type == "prompt_start"]
    assert [c.cache for c in starts] == ["hit", "miss"]
    assert [c.prompt for c in starts] == ["2+3", "4+5"]
    assert [c.prompt_index for c in starts] == [1, 2]


def test_prompt_start_is_off_when_the_cache_is_disabled(
        tmp_path: Path) -> None:
    """Cache desactive : l'etat annonce est 'off', pas 'miss'."""
    generator = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    generator.cache_enabled = False
    generator.cache = CmmCache(
        path=tmp_path / "answers.json", enabled=False
    )
    seen: list[Any] = []
    generator.on_change = seen.append

    generator.run()

    assert seen[0].event_type == "prompt_start"
    assert seen[0].cache == "off"
    assert not (tmp_path / "answers.json").exists()


def test_a_hit_shows_the_cached_call_right_after_prompt_start(
        tmp_path: Path) -> None:
    """Un hit n'a pas de tokens : une seule ligne [CACHED] suffit a la vue."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    seen: list[Any] = []
    second.on_change = seen.append
    second.run()

    assert [c.event_type for c in seen[:2]] == ["prompt_start", "append"]
    assert seen[1].cache == "hit"
    assert "fn_add" in seen[1].text
    assert seen[1].text.endswith("[CACHED]")


def test_cached_run_returns_exactly_the_same_function_call(
        tmp_path: Path) -> None:
    """Aucune difference de resultat entre generation et hit de cache."""
    first = build_cached(
        ScriptedDecoder({"2+3": SUM_JSON}), ["2+3"], tmp_path
    )
    first.run()

    second = build_cached(ScriptedDecoder({}), ["2+3"], tmp_path)
    second.run()

    assert second.results == first.results
    fresh = first.metrics.prompts[0]
    cached = second.metrics.prompts[0]
    assert (cached.prompt, cached.name, cached.parameters) == (
        fresh.prompt, fresh.name, fresh.parameters,
    )
