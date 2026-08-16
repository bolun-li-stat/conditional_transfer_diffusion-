# Reduced ImageNet experiment plan

## Scientific role

The reduced design is an endpoint contrast intended to maximize the chance of
observing positive and negative transfer under a limited computation budget.
Close groups are selected as the conditions most favorable to positive
transfer. Far groups are extreme semantic and visual mismatches intended to
increase the chance of negative transfer. Medium groups stay within the same or
a nearby broad semantic domain and provide an intermediate transition.

Only the close and far endpoints are enabled in the current reduced grids.
Medium groups are reviewed and frozen now so that a later, finer
similarity-gradient study can reuse a predeclared design, but they create no
jobs and add no current computation. The endpoint comparison cannot establish a
monotonic relationship, threshold, or complete similarity-response curve.
Close, medium, and far are empirical design labels and are not identified with
the theoretical similarity quantity. Feature similarity remains a continuous,
independently measured diagnostic computed from the dedicated reference split.

## Frozen target and auxiliary groups

List order is fixed and contributes to the target-set hash.

| Target | Close | Medium (frozen, not run) | Far (extreme mismatch) |
|---|---|---|---|
| French bulldog (`n02108915`) | Boston bull (`n02096585`); pug (`n02110958`); boxer (`n02108089`) | tabby cat (`n02123045`); red fox (`n02119022`); timber wolf (`n02114367`) | school bus (`n04146614`); church (`n03028079`); teapot (`n04398044`) |
| jay (`n01580077`) | indigo bunting (`n01537544`); magpie (`n01582220`); chickadee (`n01592084`) | ostrich (`n01518878`); bald eagle (`n01614925`); black swan (`n01860187`) | school bus (`n04146614`); teapot (`n04398044`); mushroom (`n07734744`) |
| jeep (`n03594945`) | station wagon (`n02814533`); minivan (`n03770679`); pickup truck (`n03930630`) | airliner (`n02690373`); canoe (`n02951358`); submarine (`n04347754`) | goldfish (`n01443537`); teapot (`n04398044`); church (`n03028079`) |
| acoustic guitar (`n02676566`) | electric guitar (`n03272010`); banjo (`n02787622`); cello (`n02992211`) | accordion (`n02672831`); French horn (`n03394916`); drum (`n03249569`) | goldfish (`n01443537`); school bus (`n04146614`); mushroom (`n07734744`) |

The K-sensitivity target file uses a larger frozen French-bulldog pool:

- Close: Boston bull, pug, boxer, Chihuahua, Italian greyhound, and golden retriever.
- Medium: tabby cat, tiger cat, Egyptian cat, red fox, and timber wolf.
- Far: jeep, school bus, church, greenhouse, and mailbox.

The K-sensitivity grid likewise runs only close and far.

## Stages and static job counts

| Experiment | Role | Static jobs | Protocol breakdown |
|---|---|---:|---|
| 0: release pilot | Existing French-bulldog engineering validation and `K_aux=5` pooling stress | 12 | 6 natural, 6 target-exposure |
| 1: reduced main | Four targets, `n0={50,100,250}`, `K_aux=3`, close/far endpoints | 180 | 180 natural |
| 2: primary protocol control | French bulldog, `n0={50,100}`, ten subsets, both protocols | 120 | 60 natural, 60 target-exposure |
| 3: fixed-budget K sensitivity | French bulldog, `K_aux={1,3,5}`, total auxiliary budget 300 | 51 | 51 natural |
| **Total** |  | **363** | **297 natural, 66 target-exposure** |

Experiment 0 and the formal reduced design both use far as an extreme-mismatch
stress condition, but each uses its own predeclared and frozen class choices.
The existing release-pilot configuration, pilot target file, and readiness
status remain unchanged.

## Data and readiness boundary

The nominal 1,300 training images per class describe the standard ILSVRC2012
class allocation; they are not final evidence that a local dataset copy is
complete. Before any formal run, the existing preflight must count every
required class in the local ImageNet copy and perform its feasibility checks for
the requested holdouts, similarity references, nested target subsets, and
auxiliary budgets.

This plan records only static metadata and grid structure. It does not claim
that a real ImageNet or GPU experiment has completed, and it does not create or
alter readiness evidence.
