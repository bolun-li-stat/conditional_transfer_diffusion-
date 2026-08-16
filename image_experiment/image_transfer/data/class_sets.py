from __future__ import annotations

import hashlib
import itertools
import random

IMAGENET_CLASSES = {
    "n02108915": "French bulldog",
    "n01580077": "jay",
    "n03594945": "jeep",
    "n02676566": "acoustic guitar",
    "n02096585": "Boston bull",
    "n02110958": "pug",
    "n02108089": "boxer",
    "n02085620": "Chihuahua",
    "n02091032": "Italian greyhound",
    "n02099601": "golden retriever",
    "n02123045": "tabby cat",
    "n02123159": "tiger cat",
    "n02124075": "Egyptian cat",
    "n02119022": "red fox",
    "n02114367": "timber wolf",
    "n01537544": "indigo bunting",
    "n01582220": "magpie",
    "n01592084": "chickadee",
    "n01518878": "ostrich",
    "n01614925": "bald eagle",
    "n01860187": "black swan",
    "n02814533": "station wagon",
    "n03770679": "minivan",
    "n03930630": "pickup truck",
    "n02690373": "airliner",
    "n02951358": "canoe",
    "n04347754": "submarine",
    "n03272010": "electric guitar",
    "n02787622": "banjo",
    "n02992211": "cello",
    "n02672831": "accordion",
    "n03394916": "French horn",
    "n03249569": "drum",
    "n04146614": "school bus",
    "n03028079": "church",
    "n04398044": "teapot",
    "n07734744": "mushroom",
    "n01443537": "goldfish",
    "n03457902": "greenhouse",
    "n03710193": "mailbox",
}

CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

FRENCH_BULLDOG_AUX = {
    "close": ["n02096585", "n02110958", "n02108089", "n02085620", "n02091032", "n02099601"],
    "medium": ["n02123045", "n02123159", "n02124075", "n02119022", "n02114367"],
    "far": ["n03594945", "n04146614", "n03028079", "n03457902", "n03710193"],
}


def class_name(synset: str) -> str:
    return IMAGENET_CLASSES.get(synset, synset)


def composition_counts(name: str, k_aux: int) -> dict[str, int]:
    if name in {"close", "medium", "far"}:
        return {name: k_aux}
    if name in {"mix", "balanced_mix", "similarity_balanced_mix"}:
        if k_aux == 5:
            return {"close": 2, "medium": 1, "far": 2}
        close = round(0.4 * k_aux)
        medium = round(0.2 * k_aux)
        far = k_aux - close - medium
        return {"close": close, "medium": medium, "far": far}
    if name in {"close_only", "similarity_close_only"}:
        return {"close": k_aux}
    if name in {"far_only", "similarity_far_only"}:
        return {"far": k_aux}
    if name in {"mostly_close", "similarity_mostly_close"}:
        far = max(1, round(0.2 * k_aux)) if k_aux else 0
        return {"close": k_aux - far, "far": far}
    if name in {"mostly_far", "similarity_mostly_far"}:
        close = max(1, round(0.2 * k_aux)) if k_aux else 0
        return {"close": close, "far": k_aux - close}
    raise ValueError(f"Unknown auxiliary composition: {name}")


def _candidate_combinations(auxiliary_sets: dict[str, list[str]], composition: str, k_aux: int) -> list[list[str]]:
    """Enumerate all valid, within-draw unique auxiliary class sets."""

    group_options: list[list[tuple[str, ...]]] = []
    for group, count in composition_counts(composition, k_aux).items():
        candidates = list(dict.fromkeys(auxiliary_sets.get(group, [])))
        if len(candidates) < count:
            raise ValueError(f"Composition {composition} needs {count} {group} classes, found {len(candidates)}")
        group_options.append(list(itertools.combinations(candidates, count)))
    combinations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for grouped in itertools.product(*group_options):
        selected = [synset for group in grouped for synset in group]
        if len(selected) != k_aux or len(set(selected)) != k_aux:
            continue
        key = tuple(selected)
        if key not in seen:
            seen.add(key)
            combinations.append(selected)
    if not combinations:
        raise ValueError(f"Composition {composition} has no valid unique {k_aux}-class draw")
    return combinations


def draw_aux_synset_combinations(
    auxiliary_sets: dict[str, list[str]],
    composition: str,
    k_aux: int,
    *,
    num_draws: int,
    aux_draw_seed: int,
) -> tuple[list[list[str]], int]:
    """Return reproducibly shuffled unique draws and the number available.

    If the candidate pool permits fewer unique sets than requested, only the
    actual unique sets are returned.  Callers can record ``available`` instead
    of pretending that repeated rows are independent auxiliary draws.
    """

    candidates = _candidate_combinations(auxiliary_sets, composition, k_aux)
    digest = hashlib.sha256(f"{aux_draw_seed}\0{composition}\0{k_aux}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(candidates)
    requested = max(int(num_draws), 0)
    return candidates[: min(requested, len(candidates))], len(candidates)


def select_aux_synsets(
    auxiliary_sets: dict[str, list[str]],
    composition: str,
    k_aux: int,
    *,
    aux_draw_seed: int | None = None,
    aux_draw_id: int = 0,
) -> list[str]:
    """Select one auxiliary set, retaining the historical first-K default.

    Supplying ``aux_draw_seed`` activates reproducible random draws.  Distinct
    ``aux_draw_id`` values index a shuffled list of unique combinations.
    """

    if aux_draw_seed is not None:
        draws, available = draw_aux_synset_combinations(
            auxiliary_sets,
            composition,
            k_aux,
            num_draws=int(aux_draw_id) + 1,
            aux_draw_seed=int(aux_draw_seed),
        )
        if int(aux_draw_id) >= len(draws):
            raise ValueError(
                f"Auxiliary draw {aux_draw_id} unavailable for {composition}; only {available} unique combinations exist"
            )
        return draws[int(aux_draw_id)]

    selected: list[str] = []
    for group, count in composition_counts(composition, k_aux).items():
        candidates = auxiliary_sets.get(group, [])
        if len(candidates) < count:
            raise ValueError(f"Composition {composition} needs {count} {group} classes, found {len(candidates)}")
        selected.extend(candidates[:count])
    if len(selected) != k_aux:
        raise ValueError(f"Composition {composition} produced {len(selected)} classes, expected {k_aux}")
    return selected
