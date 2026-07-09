from __future__ import annotations

IMAGENET_CLASSES = {
    "n02108915": "French bulldog",
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
    "n03594945": "jeep",
    "n04146614": "school bus",
    "n03028079": "church",
    "n03457902": "greenhouse",
    "n03710193": "mailbox",
    "n01580077": "jay",
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


def select_aux_synsets(auxiliary_sets: dict[str, list[str]], composition: str, k_aux: int) -> list[str]:
    selected: list[str] = []
    for group, count in composition_counts(composition, k_aux).items():
        candidates = auxiliary_sets.get(group, [])
        if len(candidates) < count:
            raise ValueError(f"Composition {composition} needs {count} {group} classes, found {len(candidates)}")
        selected.extend(candidates[:count])
    if len(selected) != k_aux:
        raise ValueError(f"Composition {composition} produced {len(selected)} classes, expected {k_aux}")
    return selected
