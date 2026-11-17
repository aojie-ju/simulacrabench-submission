"""SimulacraBench submission, v13: v12, with items gated by the same parent on the same
answers choosing their gate parent together.

  - Siblings (same parent, same answers that open the gate: in UNHCR the return,
    do-not-return and undecided branches off s3_11, 73 items in all; in World Bank the
    21 follow-up items gated by the feedback arm) were asked by one routing, so they
    share whatever decides who was asked. v9-v12 chose each item's gate parent (the
    named one or an alternative, _alternatives) on that item's own held-out rows, so a
    branch could follow s3_12 in one item and s3_11 in the next on noise alone. Now a
    group of two or more siblings makes one choice (_group_parent): each candidate
    parent (the named one, and the alternatives any sibling proposes) is scored by the
    sum over the siblings of each one's best held-out log loss with that parent (its
    own direct-model share and temperature), and an alternative must beat the named
    parent by ALT_HELD nats per pooled held-out row. Each sibling then keeps its own
    share and temperature with the chosen parent. TIE_GATES = False restores v12.

v12: each instrument runs the model up to PASSES times, on different held-out splits,
and returns the average of the passes' probabilities.

  - v11 showed that v7's UNHCR 0.87 was the Development noise: score.py adds Laplace
    noise before rounding (for UNHCR its sd is 12% of a display step), and v7's own
    code showed 0.85 in v11. So UNHCR runs this file's code again, like the others,
    and the bundled unhcr_v7.py is gone.
  - Every choice inside a pass (strengths, blend weights, temperatures, gate chains,
    alternative parents) is made on one held-out split of about a fifth of the
    visible rows, so each carries selection noise. Passes differ in the split (seed
    SEED + 7919 * pass), in TabPFN's seed and context rows, and in the multi-task
    members' seeds; their vectors are averaged with equal weights. The log score is
    concave in the probability, so the average scores at least as well as the passes
    do on average (Jensen), and usually better than any one of them.
  - Time. A pass keeps Development's timing: its deadlines (trees and TabPFN 700 s,
    direct models 780 s, tables 820 s, the multi-task model 700 s; TabPFN waits 25 /
    100 / 200 s) count from the pass's own start, but never fall later than offset *
    SCALE after import, so with BUDGET = 900 they are v11's. Another pass starts only
    while PASS_ROOM times the first pass's duration fits before the earlier of the
    tree deadline (TREE_DEADLINE * SCALE after import) and the instrument's share of
    the budget: its cost in a hosted Development pass (COSTS: UNHCR 410 s, UNICEF
    25 s, World Bank 140 s) over the cost of the instruments not yet called, times
    the budget left (BUDGET - MARGIN after import), so it holds in any order. With
    BUDGET = 900 that is one pass for UNHCR and World Bank and two for UNICEF; with
    3600 about four for UNHCR and five for the others.

v11: the UNHCR call ran v7's code; UNICEF and World Bank ran v10's, with deadlines set
for the order the hosted run really uses.

  - The hosted run calls the instruments in alphabetical order: UNHCR first, then
    UNICEF, then World Bank. probe-timing-v10 showed UNHCR's predict() starting within
    20 s of import and ending 400-420 s after it (and TabPFN calls of 0.8 s on the
    GPU). Notes below that say UNHCR runs last are wrong. So was MT_DEADLINE: 300 s
    after import had passed before UNICEF and World Bank started, so neither has had
    the multi-task model in any hosted run since v5. It now runs until 700 s, trees
    and TabPFN may start until 700 s, direct models until 780 s and country tables
    until 820 s; World Bank, last, starts at about 450-500 s.
  - UNHCR scored 0.87 on Development with v7 and 0.85 with v8 and v10 (one display
    step, 0.019 of skill). The step came with v8, whose TabPFN thread v9 and v10
    keep, so the UNHCR call runs v7's main.py unchanged (unhcr_v7.py, commit
    dec6554). It is called first, as it was for v7, so its clock is v7's too. If
    that module fails to import or raises, UNHCR falls back to the code in this file.
  - scripts/make_test_build.py patches BUDGET, TABPFN_ESTIMATORS and TABPFN_REFIT in
    both files.

v10: v9 plus an ordered logistic model for items whose options are ordered numbers.

  - World Bank trails the board by one display step (0.23 against 0.25), and 51 of its
    76 PREDICT items have at least four numeric options: scale scores in half-point
    bands, test scores, hours and applications per week, wage bands. The multinomial
    logistic model gives every option its own slope on every GIVEN column, so a rare
    band ("31-40 hours") is estimated from its own few rows. For these items the
    logistic slot now also tries a proportional-odds model (cumulative logit): one
    slope per GIVEN column shared by all the ordered options, and cutpoints between
    them, so a rare band borrows the shift its neighbours show. Options that are not
    numbers ("Not answered", "never asked") get a multinomial logistic model against
    "an ordered option" as one class. Both are fit for each strength in the slot's
    grid, and the held-out rows choose among them and the multinomial model, as they
    choose its strength; the chosen one is refit on every usable row.
  - An alternative gate parent (v9) is now kept only if its mixture beats the named
    parent's by ALT_HELD nats per held-out row. On the sandbox copy whose branches follow
    s3_12, where nothing is predictable from GIVEN, v9 kept 8 alternatives, mostly an
    unrelated item: held-out noise.

v9: v8 with gate chains that read every option of the parent, and with ordered GIVEN
answers also entered as numbers.

  - Gate chains. v2-v8 gave a gated item two rates: the chance it is asked when its
    parent's answer opens the gate, and when it does not. The UNHCR schema says some of
    its gates are inferred rather than satisfied by the rows (a gate and who actually
    holds a value agree with a median of 0.83, 0.29 at worst), and that the return and
    do-not-return branches, which name s3_11 as their parent (73 items hang off it),
    follow s3_12 more closely. Now every option of the parent gets its own rate, counted
    on the visible rows and shrunk toward the two-rate value by GATE_PRIOR rows. Each
    gated item also tries as its parent up to ALT_PARENTS earlier PREDICT items whose
    answers say who was asked better than the named parent does on the fit rows (items
    gated like itself are left out: they only repeat its own gate). The chain whose
    mixture with the direct model scores best on the held-out rows is kept; a tie keeps
    the named parent.
  - Ordered GIVEN answers. Age bands, wage bands and household member counts entered
    only as unordered levels: indicators for the logistic models, the multi-task model
    and the content trees, level codes for the direct trees and TabPFN. Each GIVEN
    column with at least four numeric options now also enters as the rank of its level.
    Two such columns with the same options (World Bank's wage of similar employees and
    lowest acceptable pay) also enter as the difference of their ranks, and count
    columns enter summed (UNHCR's household members by age and sex: the household's
    size). The trees and TabPFN get the ranks (NaN where the level is not a number);
    the linear models get them standardized (0 there).

v8: v7 with TabPFN running beside the CPU models instead of before them.

  - v6 and v7 ran TabPFN on the GPU first and the CPU models after it, so each
    instrument paid for both, one after the other: up to 25 + 75 + 160 s of TabPFN
    pushed UNHCR, which runs last on the shared 900 s clock, toward the deadlines after
    which its last items lose their trees (640 s), and past the start time after which
    an instrument gets no multi-task model (300 s).
  - Now a thread feeds TabPFN on the GPU while the worker processes fit the logistic
    models, trees, country tables and blends, exactly as v7 did when TabPFN had not run
    (refitting on every visible row only what a blend uses). They return each blend
    with its candidates' held-out and target predictions, and once TabPFN has finished,
    or its time is up, the main process chooses again, with TabPFN as a fifth
    candidate, the blends whose item TabPFN finished: the same grid of weights, pooled
    share, temperature and smoothing, among the candidates that were refit. TabPFN gets
    the whole CPU stage, and at least its cap from the instrument's start, so it covers
    more items while the run gets shorter.
  - Without TabPFN the predictions are identical to v7's.

v7: v6 plus TabPFN on items with more than ten options, smoothing across ordered
options, and a temperature for the gated items' final mixture.

  - More than ten options. TabPFN v2 handles at most ten classes, so v6 left out 18
    World Bank items (sectors, courses, search hours, application counts, wage bands,
    scale scores), 2 UNHCR items (months), and the gated items whose whole distribution
    has more than ten options. Now the nine classes most frequent in the context keep
    their own class and the rest share a tenth, whose probability is split among them
    in proportion to their counts in the context.
  - Ordered options. When at least four of an item's options are numbers or numeric
    ranges ("0", "31-40", "more than 40", "25000+", "2.125"), the chosen blend gets one
    more candidate: each ordered option's ratio to the pooled distribution is pulled
    toward its neighbours' by a share s (0.1 to 0.5), chosen on the held-out rows
    together with the temperature and the pooled share; s = 0 keeps v6's blend. A
    classifier treats "31-40" and "more than 40" as unrelated classes, so the rare ones
    are estimated from few rows; this lets them borrow from their neighbours. Options
    that are not numbers ("Not answered", "never asked") keep their ratio, and so do
    codes such as 98 and 99 next to values below half of them (months 1-12).
  - The gated items' final mixture (direct model against gate chain) gets its own
    temperature, chosen on the held-out rows together with the share. A linear mixture
    of calibrated probability forecasts is underconfident (Ranjan and Gneiting 2010),
    and most UNHCR cells are near-certain. A tie keeps no sharpening.
  - BUDGET (900 s here) scales every deadline and TabPFN cap, so the Test build (3600
    s) changes one number. TABPFN_REFIT (off here, for time) predicts the target rows
    from every visible row instead of the fit rows: two TabPFN calls per item.

v6: v5.1 plus a pretrained tabular model on the GPU and calibration chosen per item.

  - TabPFN v2 (Hollmann et al., Nature 2025) enters every blend as a fifth candidate.
    It is a transformer pretrained on synthetic tabular tasks that predicts in context:
    given labeled rows it returns class probabilities for new rows without training,
    and its probabilities are well calibrated, which is what the log score rewards. It
    runs on the hosted H100 (weights from models.txt, fetched before the clock starts).
    Context: the fit rows only, so the held-out comparison stays honest. Gated items'
    whole distributions go first. Without a GPU, or on any failure, it is skipped.
  - Sharpening. After the blend is chosen, a temperature (p to the power t, then
    renormalized) is chosen on the held-out rows, with the pooled share chosen again
    for each t. The pooled share can now be 0 to 0.25 (it was at least 0.01), and the
    share of the direct model against the gate chain moves in steps of 0.1 (was 0.25).
    Most UNHCR cells are near-certain, so confidence there decides the score.

v5.1: v5 with later deadlines (trees until 640 s, direct models until 740 s, the country
table only after 780 s), and nothing else changed.

v4 scored 0.83 on UNHCR, three display steps below the top; v4.2 and v5 kept the mean at
0.47, which one UNHCR step would have lifted to 0.48. UNHCR runs last on the one 900 s
clock, and locally it finishes at 312 s on 4 cores; a hosted run
about 1.4x slower would start dropping trees from its last items at 420 s. If the hosted
run is fast enough, this version predicts exactly what v5 does; if not, UNHCR rises.

v5: v4.2 plus partial pooling across items, and a country table whose shrinkage is
chosen per item.

For each PREDICT item:

  1. The answer among respondents who were asked the item (gate open) is modeled three
     ways: v1's country table shrunk toward the pooled distribution; multinomial
     logistic regression on every GIVEN column (one-hot), its strength chosen from a
     small grid; and gradient-boosted trees, which can pick up interactions such as
     country x age. On 20% of the visible respondents, held out at random, the blend
     weights of the three and the share of the pooled distribution mixed in are chosen
     by log score; each model is then refit on all visible respondents and blended with
     those weights. The real data exists only in the hosted run, so these choices have
     to be made there, automatically.
  2. A gated item gets two candidate distributions over its answers plus "never asked".
     - The gate chain (v2, v3). With a GIVEN parent, the gate status is known. With a
       PREDICT parent, the chance the gate opens is the predicted mass on the parent's
       opening answers (parents are processed first, so chains stay consistent). The
       child gets that share of its modeled answer, the rest on "never asked".
     - Direct (v4): the same three-model blend, fit on every visible respondent with
       "never asked" as one more option, so it is predicted from the GIVEN block like
       any other answer. Its trees split on sets of levels of each GIVEN column, which
       is faster than one-hot. v4 fit it only for items with a PREDICT parent; v4.2
       fits it for items with a GIVEN parent too: World Bank (27 such items) trails the
       board, 0.23 against 0.25, and a gate that leaks in a way the GIVEN block
       predicts is exactly what the direct model can learn.
     The share of each is chosen on the held-out respondents, item by item, and the
     held-out predictions of a parent feed its children's chains, so the comparison is
     made the way the target rows are predicted.

Guards (new in v4.1; v4 finished on the hosted run without a score):
  - An item whose direct model fails keeps the gate chain (v3); an item whose blend
    fails gets the country table (v1). A failure in the held-out comparison keeps the
    chain. Neither stops the run.
  - Deadlines are absolute times set when predict() is first imported, passed to the
    worker processes (a worker that imports this module would otherwise start its own
    clock). After TABLE_DEADLINE every remaining item gets the country table only, so
    a slow machine, or the one-process fallback, still finishes inside 900 s.
  - Every returned vector is finite, non-negative and sums to 1.

Order (new in v4.2): gated items are dispatched first, and each fits its direct model
before its content models, so on a slow machine trees get cut, not direct models; a
direct model (without trees once TREE_DEADLINE has passed, which makes it cheap) is
allowed until 640 s. UNHCR runs last on the shared clock, and the direct model is what
moved it from 0.52 (v3) to 0.83 (v4).

Partial pooling (new in v5):
  - Across items. Held-out respondents show the GIVEN block only, so a respondent-level
    latent trait (item response theory) cannot be read from their own answers. What
    can be shared is how the GIVEN block maps to answers: one hidden layer, computed
    from the GIVEN block, feeds a softmax head for every item of the instrument (its
    options, plus "never asked" for a gated item). Every item trains the shared layer,
    so an item few people answered borrows the structure the others reveal: the
    multi-task form of a hierarchical model. Weight decay is a Gaussian prior, and the
    prediction averages an ensemble trained from different starting points, a cheap
    approximation to the posterior predictive. It is fit on the fit rows only and
    enters every blend as a fourth candidate: the whole distribution for a direct
    model, the answer part renormalized for the answer among those asked.
  - Within an item. The country table (v1) gave the pooled distribution the weight of
    20 respondents for every item. Now the Dirichlet-multinomial marginal likelihood
    of the item's country counts picks that weight: little pooling where countries
    really differ, much where they do not.
"""

import glob
import os
import re
import threading
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import gammaln

PRIOR = 0.5                    # pseudo-count per option in pooled distributions
SHRINK = 20.0                  # weight of the pooled distribution in a country cell (v1)
CS = (0.03, 0.1, 0.3, 1.0)     # inverse regularization strengths tried for the logistic model
TREES = dict(learning_rate=0.15, max_leaf_nodes=15, min_samples_leaf=40,
             l2_regularization=1.0)
TREE_ITERATIONS = 60           # upper bound; early stopping picks the number on the fit rows
MIN_TREE_ROWS = 300            # fewer fit rows than this: no trees for the model
BUDGET = 3600                  # seconds for all three instruments together (Test phase)
SCALE = BUDGET / 900           # v12: only MARGIN scales; a pass keeps Development's timing
TREE_DEADLINE = 700.0          # seconds after a pass starts; later models skip trees
DIRECT_DEADLINE = 780.0        # seconds after a pass starts; later gated items keep the chain
                               # (after TREE_DEADLINE a direct model has no trees: cheap)
TABLE_DEADLINE = 820.0         # seconds after a pass starts; later items get the country table
DIRECT_CS = (0.1, 1.0)         # logistic strengths tried for the direct model (to save time)
MAX_CATEGORIES = 254           # a GIVEN column with more levels enters the trees as a number
START = time.time()
PASSES = 5                     # most passes per instrument, on different held-out splits
PASS_ROOM = 1.5                # another pass starts only if this many first passes still fit
MARGIN = 75.0 * SCALE          # seconds of the budget kept free at the end
UNHCR_ITEM = "s1_2_country_asylum_lbl"  # an item only the UNHCR schema has
WB_ITEM = "sc_nstd_extra"               # an item only the World Bank schema has
COSTS = {UNHCR_ITEM: 410.0, None: 25.0, WB_ITEM: 140.0}  # seconds of one hosted Development
                               # pass (probe-timing-v10 and v11); None: UNICEF or any other
_SEEN = set()                  # instruments already called in this process
PASS_LOG = []                  # (instrument item, passes, first pass s), for local checks
GRID = (0, .25, .5, .75, 1)
WEIGHTS = tuple((a, b, c, d, round(1 - a - b - c - d, 2))
                for a in GRID for b in GRID for c in GRID for d in GRID
                if a + b + c + d <= 1)    # logit, trees, table, multi-task model, TabPFN
ALPHAS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0)  # pooled weight, respondents
MIXES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25)  # pooled share mixed into the blend
TAUS = (1.0, 0.8, 1.25, 1.5, 2.0, 3.0)  # sharpening (1: none; first, so a tie keeps it)
SMOOTHS = (0.1, 0.2, 0.35, 0.5)  # pull of an ordered option's ratio toward its neighbours'
SHARES = tuple(j / 10 for j in range(11))  # weight of the direct model against the chain
GATE_PRIOR = 10.0              # rows' worth of the two-rate value behind each option's rate
ALT_PARENTS = 3                # most alternative parents tried per gated item
ALT_MARGIN = 0.005             # nats per fit row an alternative must gain over the named parent
ALT_HELD = 0.002               # v10: nats per held-out row it must then gain to be kept
TIE_GATES = True               # v13: siblings choose their gate parent together
FREE = object()                # _final's parent_choice: choose for this item alone
MAX_COUNT = 30                 # count columns: no numeric option above this
TABPFN_DEADLINE = 700.0        # seconds after a pass starts; no TabPFN fit starts later
TABPFN_CAPS = ((20, 25.0), (100, 100.0), (10 ** 9, 200.0))
                               # (most items, seconds from the instrument's start): how
                               # long blending waits for TabPFN when the CPU models finish
                               # first; TabPFN runs beside them in any case
TABPFN_GRACE = 30.0            # seconds to wait for the TabPFN call in progress at the end
TABPFN_CONTEXT = 10000         # most rows given to TabPFN as labeled context
TABPFN_ESTIMATORS = 4          # TabPFN's own ensemble over preprocessings (v12: 4 in Test)
TABPFN_REFIT = False           # True: the target rows get every visible row as context
TABPFN_MAX_CLASSES = 10        # TabPFN v2 handles at most 10 classes; more are merged
TABPFN_MIN_ROWS = 50           # fewer context rows than this: skip
TABPFN_REPO = "models--Prior-Labs--TabPFN-v2-clf"   # the models.txt entry, cache form
TABPFN_FILE = "tabpfn-v2-classifier.ckpt"
HOLDOUT = 0.2                  # share of visible respondents held out to choose a model
MIN_VALIDATION = 30            # fewer usable validation rows than this: keep the default
FLOOR = 0.001                  # the grader's floor, used when scoring candidates
SEED = 20261021
N_JOBS = 8
MT_HIDDEN = 32                 # width of the layer shared by all items
MT_DECAY = 1e-4               # weight decay: a Gaussian prior on the weights
MT_RATE = 0.05                # Adam step size
MT_STEPS = 200                # upper bound; early stopping picks the number
MT_CHECK = 10                 # steps between early-stopping checks
MT_PATIENCE = 20              # steps without improvement before stopping
MT_VALIDATION = 0.15          # share of the fit rows used for early stopping
MT_SEEDS = (1, 2)             # ensemble members
MT_DEADLINE = 700.0           # seconds after a pass starts; later passes skip it (v10: 300 s
                              # after import, which UNICEF and World Bank never met)
NUMBER = re.compile(r"(?:(more than|greater than|over|above|after|at least|>=|>|\u2265|"
                    r"less than|fewer than|under|below|before|<=|<|\u2264) ?)?"
                    r"(\d+(?:\.\d+)?)(?: ?(\+|or more|and above|and over)"
                    r"| ?[-\u2013] ?\d+(?:\.\d+)?)?")
ABOVE = {"more than", "greater than", "over", "above", "after", "at least", ">=", ">",
         "\u2265"}
CODES = {96.0, 97.0, 98.0, 99.0, 996.0, 997.0, 998.0, 999.0,
         9996.0, 9997.0, 9998.0, 9999.0}  # "don't know"-style codes beside small numbers


def predict(frame, schema):
    """Up to PASSES runs of _predict on different held-out splits, averaged (v12). The
    instrument may use its share of the budget left (see the docstring)."""
    items = schema["items"]
    key = UNHCR_ITEM if UNHCR_ITEM in items else WB_ITEM if WB_ITEM in items else None
    left = [k for k in COSTS if k not in _SEEN or k == key]
    _SEEN.add(key)
    now = time.time()
    share = COSTS[key] / sum(COSTS[k] for k in left)
    end = now + share * max(START + BUDGET - MARGIN - now, 0.0)
    runs, first = [], None
    for p in range(PASSES):
        began = time.time()
        if p and began + PASS_ROOM * first > min(end, START + TREE_DEADLINE * SCALE):
            break
        try:
            runs.append(_predict(frame, schema, SEED + 7919 * p, began, end))
        except Exception:
            if not runs:
                raise
            break
        if first is None:
            first = time.time() - began
    PASS_LOG.append((key, len(runs), round(first or 0.0, 1)))
    return _average(runs)


def _average(runs):
    """The passes' vectors, each normalized, averaged cell by cell."""
    if len(runs) == 1:
        return runs[0]
    out = []
    for cells in zip(*runs):
        total = np.zeros(len(cells[0]))
        for vector in cells:
            vector = np.asarray(vector, dtype=float)
            total += vector / vector.sum()
        out.append(total / len(cells))
    return out


def _predict(frame, schema, seed=SEED, origin=None, end=np.inf):
    """One pass of the model (v10's, with v11's deadlines): the held-out split, TabPFN
    and the multi-task members are seeded by `seed`, and every deadline counts from
    `origin` (by default the import), never later than its offset times SCALE after
    the import. `end` (the instrument's share) only decides whether another pass
    starts, in predict()."""
    origin = START if origin is None else origin

    def at(offset):
        return min(origin + offset, START + offset * SCALE)

    items = schema["items"]
    gated_value = schema["gated_value"]
    shipped = [name for name, record in items.items()
               if record["class"] in ("GIVEN", "PREDICT")]
    position = {name: k for k, name in enumerate(shipped)}
    values = frame[shipped].to_numpy(dtype=object)
    blank = pd.isna(values)
    visible = ~blank.any(axis=1)
    targets = np.flatnonzero(blank.any(axis=1))

    given = [name for name in shipped if items[name]["class"] == "GIVEN"]
    columns = [position[name] for name in given]
    features = _one_hot(values, columns, blank)
    try:
        ranks, scaled = _numbers(values, columns, [items[name]["values"] for name in given],
                                 blank)
    except Exception:
        ranks = scaled = np.zeros((len(values), 0), dtype=np.float32)
    features = np.hstack([features, scaled]).astype(np.float32)
    try:
        categories, categorical = _codes(values, columns, blank)
        categories = np.hstack([categories, ranks]).astype(np.float32)
        categorical = np.concatenate([categorical, np.zeros(ranks.shape[1], dtype=bool)])
    except Exception:
        categories, categorical = features, None
    country = next((name for name in given if "country" in name.lower()), None)
    country_id = (np.zeros(len(values), dtype=int) if country is None else
                  pd.factorize(values[:, position[country]], use_na_sentinel=False)[0])

    rng = np.random.default_rng(seed)
    seen_rows = np.flatnonzero(visible)
    held = rng.random(len(seen_rows)) < HOLDOUT
    split = (seen_rows[~held], seen_rows[held])

    to_model = [name for name in shipped if blank[:, position[name]].any()]
    inputs = (features, categories, categorical, country_id)
    clock = (at(TREE_DEADLINE), at(DIRECT_DEADLINE), at(TABLE_DEADLINE))
    tabpfn_until = at(TABPFN_DEADLINE)
    # TabPFN runs on the GPU in a thread beside everything below; its results are read
    # only once the CPU models are done (see _tabpfn for the order of its items).
    tab, lock, stop, worker = {}, threading.Lock(), threading.Event(), None
    began = time.time()
    wait_until = began
    if to_model and began < tabpfn_until:
        cap = next(seconds for most, seconds in TABPFN_CAPS if len(to_model) <= most)
        wait_until = min(began + cap, tabpfn_until)
        try:
            worker = threading.Thread(
                target=_tabpfn_quietly, daemon=True,
                args=(values, position, items, to_model, gated_value, visible, split,
                      targets, categories, categorical, tabpfn_until, tab, lock,
                      stop, seed))
            worker.start()
        except Exception:
            worker = None
    multi = {}
    if to_model and time.time() < at(MT_DEADLINE):
        try:
            multi = _multitask(features, values, position, items, to_model, gated_value,
                               split, targets, tuple(s + seed - SEED for s in MT_SEEDS))
        except Exception:
            multi = {}
    jobs = [(items[name], bool(items[name].get("gate")), gated_value,
             values[:, position[name]], inputs, visible, split, targets, clock,
             multi.get(name))
            for name in to_model]
    # Gated items first: their direct models are worth the most, and on a slow machine
    # the items dispatched last are the ones that lose trees.
    order = sorted(range(len(jobs)), key=lambda j: not jobs[j][1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fitted = Parallel(n_jobs=N_JOBS)(delayed(_models)(*jobs[j]) for j in order)
        except Exception:
            # The hosted container returns no traceback, so a failure to start worker
            # processes would cost a whole day. Run the same work in this process instead;
            # the deadlines above still hold.
            fitted = [_models(*jobs[j]) for j in order]
    candidates = {to_model[j]: result for j, result in zip(order, fitted)}
    if worker is not None:
        worker.join(max(0.0, wait_until - time.time()))
        stop.set()
        worker.join(TABPFN_GRACE)          # the call in progress; a later result is dropped
    with lock:
        stop.set()
        tab = {name: dict(kinds) for name, kinds in tab.items()}
    model = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in to_model:
            model[name] = _choose(items[name], bool(items[name].get("gate")),
                                  candidates[name], multi.get(name), tab.get(name))

    fit = np.zeros(len(values), dtype=bool)
    fit[split[0]] = True
    final, final_held, cache = {}, {}, {}
    groups = _sibling_groups(items, to_model) if TIE_GATES else {}
    decided = {}
    for name in _parents_first(items, to_model):
        key = _gate_key(items[name])
        if key in groups and key not in decided:
            try:
                decided[key] = _group_parent(items, groups[key], gated_value, values,
                                             position, fit, split[1], model, final,
                                             final_held, cache)
            except Exception:
                decided[key] = FREE
        final[name], final_held[name] = _final(items, name, gated_value, values, position,
                                               visible, fit, split[1], targets,
                                               model[name], final, final_held, cache,
                                               decided.get(key, FREE))

    valid = {name: _valid(p) for name, p in final.items()}
    row_of = {row: k for k, row in enumerate(targets)}
    vectors = []
    for row in targets:
        for column in np.flatnonzero(blank[row]):
            vectors.append(valid[shipped[column]][row_of[row]])
    return vectors


def _one_hot(values, columns, blank):
    """Every GIVEN column as indicator columns; a missing value is a level of its own."""
    blocks = []
    for column in columns:
        labels = np.where(blank[:, column], "<missing>", values[:, column].astype(str))
        codes, levels = pd.factorize(labels)
        block = np.zeros((len(values), len(levels)), dtype=np.float32)
        block[np.arange(len(values)), codes] = 1.0
        blocks.append(block[:, 1:] if len(levels) > 1 else block[:, :0])
    return np.hstack(blocks) if blocks else np.zeros((len(values), 0), dtype=np.float32)


def _numbers(values, columns, options, blank):
    """The ordered GIVEN columns as numbers (v9): the rank of the level for each column
    with at least four numeric options (_ordinal; a blank or non-numeric level is NaN),
    the difference of the ranks of two such columns with the same options, and the sum
    of the count columns when there are at least two (options that are whole numbers or
    ranges of them, none above MAX_COUNT: "6-10" counts as 8, "11+" as 11). Returns
    (the columns with NaN, for the trees and TabPFN; the same standardized with NaN at
    0, for the logistic and multi-task models)."""
    ranks, counts, same = [], [], {}
    for column, levels in zip(columns, options):
        levels = list(levels)
        order = _ordered(levels)
        if order is None:
            continue
        labels = np.where(blank[:, column], None, values[:, column])
        rank = {str(levels[j]): float(r) for r, j in enumerate(order)}
        ranks.append(np.array([rank.get(str(v), np.nan) if v is not None else np.nan
                               for v in labels]))
        same.setdefault(tuple(map(str, levels)), []).append(len(ranks) - 1)
        amount = _counts([levels[j] for j in order])
        if amount is not None:
            value = {str(levels[j]): a for j, a in zip(order, amount)}
            counts.append(np.array([value.get(str(v), np.nan) if v is not None else np.nan
                                    for v in labels]))
    out = list(ranks)
    for members in same.values():
        if len(members) == 2:
            out.append(ranks[members[1]] - ranks[members[0]])
    if len(counts) >= 2:
        stack = np.column_stack(counts)
        total = np.nansum(stack, axis=1)
        total[np.isnan(stack).all(axis=1)] = np.nan
        out.append(total)
    if not out:
        empty = np.zeros((len(values), 0), dtype=np.float32)
        return empty, empty
    raw = np.column_stack(out).astype(np.float64)
    seen = ~np.isnan(raw)
    mean = np.array([raw[seen[:, j], j].mean() if seen[:, j].any() else 0.0
                     for j in range(raw.shape[1])])
    spread = np.array([raw[seen[:, j], j].std() if seen[:, j].any() else 1.0
                       for j in range(raw.shape[1])])
    spread[~(spread > 0)] = 1.0
    scaled = np.where(seen, (raw - mean) / spread, 0.0)
    return raw.astype(np.float32), scaled.astype(np.float32)


def _counts(labels):
    """The number each ordered option stands for, if all are whole numbers or ranges of
    them ("3", "6-10" as 8, "11+" as 11) and none is above MAX_COUNT; else None."""
    amounts = []
    for label in labels:
        text = " ".join(str(label).lower().split())
        match = NUMBER.fullmatch(text)
        if match is None or "." in match.group(2):
            return None
        low = float(match.group(2))
        high = re.findall(r"\d+(?:\.\d+)?", text)
        value = (low + float(high[1])) / 2 if len(high) == 2 and not match.group(3) else low
        if low > MAX_COUNT:
            return None
        amounts.append(value)
    return amounts


def _codes(values, columns, blank):
    """Every GIVEN column as one column of level codes, for trees that split on sets of
    levels; the mask says which columns can be treated as categorical."""
    if not columns:
        return np.zeros((len(values), 0), dtype=np.float32), np.zeros(0, dtype=bool)
    codes, levels = [], []
    for column in columns:
        labels = np.where(blank[:, column], "<missing>", values[:, column].astype(str))
        code, found = pd.factorize(labels)
        codes.append(code)
        levels.append(len(found))
    return (np.column_stack(codes).astype(np.float32),
            np.array(levels) <= MAX_CATEGORIES)


def _gated(answers, gated_value):
    """True where the answer is the "never asked" value. Anything else, a blank
    included, is False."""
    return np.array([isinstance(answer, str) and answer == gated_value
                     for answer in answers], dtype=bool)


def _models(record, gated, gated_value, answers, inputs, visible, split, targets, clock,
            multi=None):
    """One item's blends, fit in a worker process before TabPFN's results are known
    (see _fit): for the answer among respondents who were asked the item, and, for a
    gated item, for the whole distribution with "never asked" as its last option (None
    for an ungated item, after the deadline, or if it fails). The direct model is fit
    first. An answer whose blend fails falls back to the country table alone (v1's
    model, the third value). `multi`: the multi-task model's (target rows, held-out
    rows), or None. Each blend comes back with its candidates, so that the main process
    can choose it again with TabPFN (_choose)."""
    tree_until, direct_until, table_after = clock
    features, categories, categorical, country_id = inputs
    options = list(record["values"])
    index = {option: j for j, option in enumerate(options)}
    code = np.array([index.get(answer, -1) for answer in answers])
    usable = visible & (code >= 0)
    k = len(options)
    ordered = _ordered(options)
    direct_extra, content_extra = _extras(multi, gated, k)
    direct = None
    if gated and time.time() < direct_until:
        try:
            whole = np.where(_gated(answers, gated_value), k, code)
            direct = _fit(whole, k + 1, visible & (whole >= 0), split, targets,
                          (features, categories, categorical, country_id), DIRECT_CS,
                          tree_until, direct_extra, ordered)
        except Exception:
            direct = None
    content = None
    if time.time() < table_after:
        try:
            content = _fit(code, k, usable, split, targets,
                           (features, features, None, country_id), CS, tree_until,
                           content_extra, ordered)
        except Exception:
            content = None
    fallback = None
    if content is None:
        fallback = _tables(code, k, usable, split, targets, country_id)
    return content, direct, fallback


def _ordered(options):
    """_ordinal, or None if it fails."""
    try:
        return _ordinal(list(options))
    except Exception:
        return None


def _extras(multi, gated, k):
    """The multi-task model's predictions, given as (target rows, held-out rows) of the
    whole distribution, as the (held-out rows, target rows) of a gated item's direct
    model and of its answer among those asked (renormalized); (None, None) without."""
    if multi is None:
        return None, None
    target_whole, held_whole = multi
    if gated:
        return ((held_whole, target_whole),
                (_answers_only(held_whole, k), _answers_only(target_whole, k)))
    return None, (held_whole, target_whole)


def _choose(record, gated, fitted, multi=None, tab=None):
    """One item's distributions, in the main process once TabPFN is done: the workers'
    blends (_models), each chosen again with TabPFN as a fifth candidate where TabPFN
    finished it. Returns ((target rows, held-out rows) of the answer among those asked,
    and the same for the whole distribution of a gated item, or None), as v7's _models
    did. `tab`: TabPFN's {"direct"/"content": (held-out rows, target rows)}, or None."""
    content, direct, fallback = fitted
    k = len(record["values"])
    direct_tab = content_tab = None
    if tab:
        direct_tab, content_tab = tab.get("direct"), tab.get("content")
        if gated and content_tab is None and direct_tab is not None:
            content_tab = (_answers_only(direct_tab[0], k), _answers_only(direct_tab[1], k))
    ordered = _ordered(record["values"])
    direct_extra, content_extra = _extras(multi, gated, k)
    whole = None if direct is None else _with_tabpfn(direct, direct_extra, direct_tab,
                                                      ordered)
    answer = fallback if content is None else _with_tabpfn(content, content_extra,
                                                           content_tab, ordered)
    return answer, whole


def _with_tabpfn(fitted, extra, tab, ordered):
    """A worker's blend (_fit) chosen again with TabPFN's (held-out rows, target rows)
    among its candidates, leaving out those the blend without TabPFN gave no weight
    (their target rows were not refit). Without TabPFN, or on failure, the worker's own
    blend."""
    if tab is None or not fitted["valid"] or fitted["k"] == 1:
        return fitted["result"]
    try:
        params, held_out = _select(fitted, extra, tab, ordered, refit_only=True)
        return _compose(fitted, params, extra, tab, ordered), held_out
    except Exception:
        return fitted["result"]


def _fit(code, k, usable, split, targets, inputs, cs, tree_until, extra=None,
         ordered=None):
    """P(code | GIVEN) for the target rows and for every held-out row, blended as v7 did
    when TabPFN had not run (_select), plus the candidates for choosing the blend again
    with TabPFN.

    The candidates are the logistic model (its strength chosen on the held-out rows),
    the trees (iterations chosen by early stopping on the fit rows), the country table
    shrunk toward the pooled distribution, and `extra`, the multi-task model's
    (held-out rows, target rows), or None. Held-out rows get each as fit on the fit
    rows; target rows get the chosen ones refit on every usable row. `inputs`: the
    logistic model's features, the trees' features and their categorical mask (or
    None), and each row's country. `ordered`: as for _select.

    Returns a dict: "result", (target rows, held-out rows); the candidates "logit",
    "trees" and "table", each (held-out rows, target rows or None where the blend gave
    it no weight); and what _select needs."""
    features, tree_x, categorical, country_id = inputs
    held_rows = split[1]
    fit_rows = split[0][usable[split[0]]]
    check = usable[held_rows]
    all_rows = np.flatnonzero(usable)
    out = dict(k=k, n_targets=len(targets), n_held=len(held_rows), check=check,
               truth=code[held_rows[check]], valid=False,
               single=len(np.unique(code[all_rows])) < 2)
    if k == 1:
        out["result"] = (np.ones((len(targets), 1)), np.ones((len(held_rows), 1)))
        return out
    pooled = _pooled(code, k, usable)
    out["pooled"] = pooled
    alpha = _alpha(code, fit_rows, country_id, k, pooled)
    out["table"] = (_table(code, fit_rows, held_rows, country_id, k, pooled, alpha),
                    _table(code, all_rows, targets, country_id, k, pooled,
                           _alpha(code, all_rows, country_id, k, pooled)))
    c = iterations = None
    if check.sum() >= MIN_VALIDATION and len(np.unique(code[fit_rows])) > 1:
        truth = out["truth"]
        logit, best = None, np.inf
        for strength in cs:
            p = _logit(code, fit_rows, held_rows, features, k, strength, pooled)
            score = _score(p[check], truth, k)
            if score < best:
                best, c, logit = score, strength, p
        if ordered is not None:            # v10: the ordered model competes for the slot
            for strength in cs:
                try:
                    p = _cumulative(code, fit_rows, held_rows, features, k, ordered,
                                    strength, pooled)
                    score = _score(p[check], truth, k)
                except Exception:
                    continue
                if np.isfinite(score) and score < best:
                    best, c, logit = score, ("ordered", strength), p
        out["logit"] = (logit, None)
        if len(fit_rows) >= MIN_TREE_ROWS and time.time() < tree_until:
            held, iterations = _trees(code, fit_rows, held_rows, tree_x, categorical, k,
                                      None, pooled)
            out["trees"] = (held, None)
        out["valid"] = True
    params, held_out = _select(out, extra, None, ordered)
    weights = params[0]
    if weights[0] > 0:
        out["logit"] = (out["logit"][0],
                        _linear(code, all_rows, targets, features, k, c, pooled, ordered))
    if weights[1] > 0:
        out["trees"] = (out["trees"][0],
                        _trees(code, all_rows, targets, tree_x, categorical, k, iterations,
                               pooled)[0])
    out["result"] = (_compose(out, params, extra, None, ordered), held_out)
    return out


def _select(fitted, extra=None, extra2=None, ordered=None, refit_only=False):
    """The blend of one item's candidates (_fit) chosen on the held-out rows with a usable
    code: the weights of the logistic model, the trees, the country table, `extra` (the
    multi-task model) and `extra2` (TabPFN), each (held-out rows, target rows) or None,
    and the share of the pooled distribution mixed in; then a sharpening temperature,
    with the pooled share chosen again for each. `ordered`: the indices of the options
    that are numbers, in numeric order, or None; with it, smoothing across neighbours is
    tried after the temperature (each strength with every temperature and pooled share).
    `refit_only`: leave out the candidates whose target rows were not refit. Returns
    ((weights, pooled share, temperature, smoothing), the held-out rows' distribution)."""
    k = fitted["k"]
    pooled, check = fitted["pooled"], fitted["check"]
    weights, mix = (0.0, 0.0, 1.0, 0.0, 0.0), 0.0  # table
    tau, smooth = 1.0, 0.0
    held_out = fitted["table"][0]
    if fitted["valid"]:
        truth = fitted["truth"]
        table = held_out
        logit = fitted["logit"][0]
        use_trees = "trees" in fitted
        trees = fitted["trees"][0] if use_trees else 0.0
        multi = extra[0] if extra is not None else 0.0
        tab = extra2[0] if extra2 is not None else 0.0
        missing = [False, not use_trees, False, extra is None, extra2 is None]
        if refit_only:
            missing[0] = fitted["logit"][1] is None
            missing[1] = missing[1] or fitted["trees"][1] is None
        best = np.inf
        for w in WEIGHTS:
            if any(share and gone for share, gone in zip(w, missing)):
                continue
            blend = w[0] * logit + w[1] * trees + w[2] * table + w[3] * multi + w[4] * tab
            for m in MIXES:
                p = (1 - m) * blend + m * pooled
                score = _score(p[check], truth, k)
                if score < best:
                    best, weights, mix, held_out = score, w, m, p
        w = weights
        blend = w[0] * logit + w[1] * trees + w[2] * table + w[3] * multi + w[4] * tab
        for t in TAUS:
            if t == 1.0:
                continue
            sharp = _sharpen(blend, t)
            for m in MIXES:
                p = (1 - m) * sharp + m * pooled
                score = _score(p[check], truth, k)
                if score < best:
                    best, mix, tau, held_out = score, m, t, p
        if ordered is not None:
            for s in SMOOTHS:
                smoothed = _smooth(blend, pooled, ordered, s)
                for t in TAUS:
                    sharp = _sharpen(smoothed, t)
                    for m in MIXES:
                        p = (1 - m) * sharp + m * pooled
                        score = _score(p[check], truth, k)
                        if score < best:
                            best, mix, tau, smooth, held_out = score, m, t, s, p
    if fitted["single"]:
        weights, tau, smooth = (0.0, 0.0, 1.0, 0.0, 0.0), 1.0, 0.0
    return (weights, mix, tau, smooth), held_out


def _compose(fitted, params, extra=None, extra2=None, ordered=None):
    """The target rows' distribution for the blend `params` chosen by _select."""
    weights, mix, tau, smooth = params
    pooled = fitted["pooled"]
    out = np.zeros((fitted["n_targets"], fitted["k"]))
    if weights[0] > 0:
        out += weights[0] * fitted["logit"][1]
    if weights[1] > 0:
        out += weights[1] * fitted["trees"][1]
    if weights[2] > 0:
        out += weights[2] * fitted["table"][1]
    if weights[3] > 0:
        out += weights[3] * extra[1]
    if weights[4] > 0:
        out += weights[4] * extra2[1]
    if smooth > 0:
        out = _smooth(out, pooled, ordered, smooth)
    return (1 - mix) * _sharpen(out, tau) + mix * pooled


def _smooth(p, pooled, ordered, s):
    """Pull each ordered option's ratio to the pooled distribution toward the mean ratio
    of its neighbours in `ordered` (the end options have one neighbour) by the share s;
    the other options keep theirs. Rows are renormalized; a row without mass is left
    as it is."""
    ratio = p[:, ordered] / pooled[ordered]
    near = np.empty_like(ratio)
    near[:, 1:-1] = 0.5 * (ratio[:, :-2] + ratio[:, 2:])
    near[:, 0] = ratio[:, 1]
    near[:, -1] = ratio[:, -2]
    q = p.copy()
    q[:, ordered] = ((1 - s) * ratio + s * near) * pooled[ordered]
    total = q.sum(axis=1, keepdims=True)
    return np.where(total > 0, q / np.maximum(total, 1e-300), p)


def _ordinal(options):
    """The indices of the options that are numbers or numeric ranges, in numeric order,
    or None when fewer than four are (or two share a value). "0", "31-40", "more than
    40", "25000+", "<2.5", "before 2011" and "2.125" count; "Not answered" does not,
    and neither do codes such as 98 and 99 when every other number is below half of
    them (months 1-12) or above twice them (years)."""
    keys = {}
    for j, option in enumerate(options):
        match = NUMBER.fullmatch(" ".join(str(option).lower().split()))
        if match is None:
            continue
        key = float(match.group(2))
        if match.group(3) or match.group(1) in ABOVE:
            key += 1e-6                    # "more than 40" comes after "40" and "31-40"
        elif match.group(1):
            key -= 1e-6                    # "<2.5" comes before "2.5-3.0"
        keys[j] = key
    codes = [j for j, key in keys.items() if key in CODES]
    rest = [key for key in keys.values() if key not in CODES]
    if codes and rest and (max(rest) < min(keys[j] for j in codes) / 2
                           or min(rest) > 2 * max(keys[j] for j in codes)):
        for j in codes:                    # 98 "don't know" beside months, or years
            del keys[j]
    if len(keys) < 4 or len(set(keys.values())) < len(keys):
        return None
    return np.array(sorted(keys, key=keys.get), dtype=int)


def _sharpen(p, tau):
    """p to the power tau, renormalized per row: tau > 1 makes a blend more confident,
    tau < 1 less. A row without mass is left as it is."""
    if tau == 1.0:
        return p
    q = np.power(np.maximum(p, 0.0), tau)
    total = q.sum(axis=1, keepdims=True)
    return np.where(total > 0, q / np.maximum(total, 1e-300), p)


def _tables(code, k, usable, split, targets, country_id):
    """The country table alone (v1's model): the fallback when time is short or the
    blend fails. Same shape as _select's result."""
    pooled = _pooled(code, k, usable)
    fit_rows = split[0][usable[split[0]]]
    return (_table(code, np.flatnonzero(usable), targets, country_id, k, pooled),
            _table(code, fit_rows, split[1], country_id, k, pooled))


def _pooled(code, k, usable):
    pooled = np.full(k, PRIOR)
    np.add.at(pooled, code[usable], 1)
    return pooled / pooled.sum()


def _score(p, truth, k):
    """Mean negative log score after the grader's normalization and floor."""
    p = FLOOR + (1 - k * FLOOR) * p / p.sum(axis=1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(truth)), truth]))


def _table(code, train, test, country_id, k, pooled, alpha=SHRINK):
    cells = int(country_id.max()) + 1
    counts = np.zeros((cells, k))
    np.add.at(counts, (country_id[train], code[train]), 1)
    table = (counts + alpha * pooled) / (counts.sum(axis=1, keepdims=True) + alpha)
    return table[country_id[test]]


def _alpha(code, rows, country_id, k, pooled):
    """Empirical Bayes for the country table: the weight of the pooled distribution, in
    respondents, that maximizes the Dirichlet-multinomial marginal likelihood of the
    item's country counts. With fewer than two countries, v1's fixed 20."""
    cells = int(country_id.max()) + 1
    counts = np.zeros((cells, k))
    np.add.at(counts, (country_id[rows], code[rows]), 1)
    counts = counts[counts.sum(axis=1) > 0]
    if len(counts) < 2:
        return SHRINK
    totals = counts.sum(axis=1)
    best, choice = -np.inf, SHRINK
    for alpha in ALPHAS:
        prior = alpha * pooled
        value = (np.sum(gammaln(alpha) - gammaln(totals + alpha))
                 + np.sum(gammaln(counts + prior) - gammaln(prior)))
        if value > best:
            best, choice = value, alpha
    return choice


def _answers_only(p, k):
    """The answer part of a whole distribution, renormalized: P(answer | asked)."""
    answers = p[:, :k]
    total = answers.sum(axis=1, keepdims=True)
    return np.where(total > 0, answers / np.maximum(total, 1e-12), 1.0 / k)


def _tabpfn_quietly(*args):
    """_tabpfn in the thread that predict() starts; a failure leaves what it found."""
    try:
        _tabpfn(*args)
    except Exception:
        pass


def _tabpfn(values, position, items, names, gated_value, visible, split, targets,
            categories, categorical, until, out=None, lock=None, stop=None, seed=SEED):
    """TabPFN v2's probabilities for each item: the whole distribution with "never asked"
    last for a gated item ("direct"), and the answer among those asked ("content").
    Context: the item's usable fit rows (at most TABPFN_CONTEXT); one call predicts the
    held-out rows and the target rows (with TABPFN_REFIT, a second call predicts the
    target rows from every usable visible row). More than ten classes are merged (see
    _tabpfn_fit). Gated items' whole distributions go first, then ungated items, then
    gated items' answers, each group by the entropy of its pooled answers, highest
    first; nothing starts after `until` or once `stop` is set, and an item that fails is
    skipped. Fills and returns `out`, {name: {kind: (held-out rows, target rows)}},
    holding `lock` for each write and writing nothing once `stop` is set; {} without a
    GPU."""
    out = {} if out is None else out
    lock = threading.Lock() if lock is None else lock
    stop = threading.Event() if stop is None else stop
    if categorical is None or time.time() >= until:
        return out
    path = _tabpfn_checkpoint()
    if path is None:
        return out
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    device = "cuda" if torch.cuda.is_available() else os.environ.get("SIMB_TABPFN_DEVICE")
    if not device:
        return out
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, model_path=path, device=device, n_estimators=TABPFN_ESTIMATORS,
        categorical_features_indices=[int(j) for j in np.flatnonzero(categorical)],
        ignore_pretraining_limits=True, random_state=seed)
    x = categories.astype(np.float32)
    held_rows = split[1]
    rows = np.concatenate([held_rows, targets])
    rng = np.random.default_rng(seed + 1)
    tasks = []
    for name in names:
        options = list(items[name]["values"])
        k = len(options)
        index = {option: j for j, option in enumerate(options)}
        column = values[:, position[name]]
        code = np.array([index.get(answer, -1) for answer in column])
        if items[name].get("gate"):
            whole = np.where(_gated(column, gated_value), k, code)
            tasks.append((0, -_entropy(whole[visible], k + 1), name, "direct", whole, k + 1))
            tasks.append((2, -_entropy(code[visible], k), name, "content", code, k))
        else:
            tasks.append((1, -_entropy(code[visible], k), name, "content", code, k))
    tasks.sort(key=lambda task: task[:2])
    every = np.flatnonzero(visible)
    for _, _, name, kind, code, k in tasks:
        if time.time() >= until or stop.is_set():
            break
        try:
            fit = split[0][code[split[0]] >= 0]
            if len(fit) > TABPFN_CONTEXT:
                fit = np.sort(rng.choice(fit, TABPFN_CONTEXT, replace=False))
            if len(fit) < TABPFN_MIN_ROWS or len(np.unique(code[fit])) < 2:
                continue
            if TABPFN_REFIT:
                held = _tabpfn_fit(model, x, code, fit, held_rows, k)
                context = every[code[every] >= 0]
                if len(context) > TABPFN_CONTEXT:
                    context = np.sort(rng.choice(context, TABPFN_CONTEXT, replace=False))
                target = _tabpfn_fit(model, x, code, context, targets, k)
            else:
                p = _tabpfn_fit(model, x, code, fit, rows, k)
                held, target = p[:len(held_rows)], p[len(held_rows):]
            if all(np.all(np.isfinite(q)) and np.all(q.sum(axis=1) > 0)
                   for q in (held, target)):
                with lock:
                    if not stop.is_set():
                        out.setdefault(name, {})[kind] = (held, target)
        except Exception:
            continue
    return out


def _tabpfn_fit(model, x, code, context, rows, k):
    """TabPFN's probabilities over the item's k options for `rows`, with the `context`
    rows as labeled examples. With more than TABPFN_MAX_CLASSES classes in the context,
    the nine most frequent keep their own class and the rest share the tenth, whose
    probability is split among them in proportion to their counts in the context."""
    found, counts = np.unique(code[context], return_counts=True)
    p = np.zeros((len(rows), k))
    if len(found) <= TABPFN_MAX_CLASSES:
        model.fit(x[context], code[context])
        p[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x[rows])
        return p
    order = np.argsort(-counts, kind="stable")
    last = TABPFN_MAX_CLASSES - 1
    keep, rare = found[order[:last]], found[order[last:]]
    share = counts[order[last:]] / counts[order[last:]].sum()
    label = np.full(k, last)
    label[keep] = np.arange(last)
    model.fit(x[context], label[code[context]])
    q = model.predict_proba(x[rows])
    for column, merged in enumerate(np.asarray(model.classes_, dtype=int)):
        if merged < last:
            p[:, keep[merged]] = q[:, column]
        else:
            p[:, rare] += q[:, [column]] * share
    return p


def _tabpfn_checkpoint():
    """The TabPFN v2 classifier weights in the Hugging Face cache, which the platform
    fills from models.txt before the container starts (there is no network at run
    time). None if they are not there."""
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HUGGINGFACE_HUB_CACHE"),
             os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    for root in roots:
        if root:
            hits = sorted(glob.glob(os.path.join(root, TABPFN_REPO, "snapshots", "*",
                                                 TABPFN_FILE)))
            if hits:
                return hits[0]
    return None


def _entropy(code, k):
    """Entropy of an item's pooled answers (nats): how much a model can gain on it."""
    counts = np.bincount(code[code >= 0], minlength=k)[:k] + PRIOR
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _multitask(features, values, position, items, names, gated_value, split, targets,
               seeds=MT_SEEDS):
    """Every item's whole distribution (its options, plus "never asked" for a gated item)
    from one hidden layer shared by all the instrument's items, averaged over an
    ensemble. Fit on the fit rows only, so its held-out predictions are honest.
    Returns {name: (target rows, held-out rows)}."""
    sizes, labels = [], []
    for name in names:
        options = list(items[name]["values"])
        gated = bool(items[name].get("gate"))
        index = {option: j for j, option in enumerate(options)}
        column = values[:, position[name]]
        code = np.array([index.get(answer, -1) for answer in column])
        if gated:
            code = np.where(_gated(column, gated_value), len(options), code)
        sizes.append(len(options) + int(gated))
        labels.append(code)
    sizes = np.array(sizes)
    labels = np.column_stack(labels)
    x = features.astype(np.float32)
    rows = np.concatenate([split[1], targets])
    prob = np.mean([_mt_fit(x, labels, sizes, split[0], rows, seed) for seed in seeds],
                   axis=0)
    held, target = prob[:len(split[1])], prob[len(split[1]):]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return {name: (target[:, s:s + n], held[:, s:s + n])
            for name, s, n in zip(names, starts, sizes)}


def _mt_fit(x, labels, sizes, train, rows, seed):
    """One ensemble member: a tanh layer shared by all items and a softmax head per
    item, trained on the mean log loss of the observed cells plus weight decay, with
    Adam and early stopping on a slice of the training rows. Every head starts at its
    item's crowd marginal. Returns probabilities for `rows`."""
    rng = np.random.default_rng(seed)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.intp)
    width = int(sizes.sum())
    stop = rng.random(len(train)) < MT_VALIDATION
    fit, val = train[~stop], train[stop]

    def onehot(subset):
        y = labels[subset]
        seen = y >= 0
        r, j = np.nonzero(seen)
        target = np.zeros((len(subset), width), np.float32)
        target[r, starts[j] + y[r, j]] = 1.0
        return (target, np.repeat(seen, sizes, axis=1).astype(np.float32),
                max(int(seen.sum()), 1))

    y_fit, mask_fit, n_fit = onehot(fit)
    y_val, _, n_val = onehot(val)
    x_fit, x_val = x[fit], x[val]
    w1 = (rng.standard_normal((x.shape[1], MT_HIDDEN))
          / np.sqrt(max(x.shape[1], 1))).astype(np.float32)
    b1 = np.zeros(MT_HIDDEN, np.float32)
    w2 = (rng.standard_normal((MT_HIDDEN, width)) * (0.1 / np.sqrt(MT_HIDDEN))).astype(np.float32)
    b2 = np.zeros(width, np.float32)
    for j, (s, n) in enumerate(zip(starts, sizes)):
        y = labels[fit, j]
        counts = np.bincount(y[y >= 0], minlength=n)[:n] + PRIOR
        b2[s:s + n] = np.log(counts / counts.sum())
    params = [w1, b1, w2, b2]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best_loss, best, stale = np.inf, [p.copy() for p in params], 0
    for step in range(1, MT_STEPS + 1):
        h = np.tanh(x_fit @ w1 + b1)
        grad = _segment_softmax(h @ w2 + b2, starts, sizes)
        grad *= mask_fit
        grad -= y_fit
        grad /= n_fit
        g_pre = (grad @ w2.T) * (1.0 - h * h)
        grads = (x_fit.T @ g_pre + MT_DECAY * w1, g_pre.sum(axis=0),
                 h.T @ grad + MT_DECAY * w2, grad.sum(axis=0))
        for p, g, m, v in zip(params, grads, first, second):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= MT_RATE * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if step % MT_CHECK == 0 and len(val):
            prob = _segment_softmax(np.tanh(x_val @ w1 + b1) @ w2 + b2, starts, sizes)
            loss = -float((y_val * np.log(prob + 1e-12)).sum()) / n_val
            if loss < best_loss - 1e-5:
                best_loss, best, stale = loss, [p.copy() for p in params], 0
            else:
                stale += MT_CHECK
                if stale >= MT_PATIENCE:
                    break
    w1, b1, w2, b2 = best
    h = np.tanh(x[rows] @ w1 + b1)
    return _segment_softmax(h @ w2 + b2, starts, sizes).astype(np.float64)


def _segment_softmax(logits, starts, sizes):
    """Softmax within each item's block of columns. The row maximum is subtracted once;
    the clip keeps a block far below it finite (it then comes out uniform)."""
    e = np.exp(np.clip(logits - logits.max(axis=1, keepdims=True), -60.0, 0.0))
    total = np.add.reduceat(e, starts, axis=1)
    return e / np.repeat(np.maximum(total, 1e-30), sizes, axis=1)


def _linear(code, train, test, features, k, c, pooled, ordered=None):
    """The logistic slot's model as chosen in _fit: multinomial for a strength c, the
    ordered model (_cumulative) for c = ("ordered", strength)."""
    if isinstance(c, tuple):
        return _cumulative(code, train, test, features, k, ordered, c[1], pooled)
    return _logit(code, train, test, features, k, c, pooled)


def _cumulative(code, train, test, features, k, ordered, strength, pooled):
    """P(code | GIVEN) for an item with ordered options (v10): which ordered option,
    among the rows that gave one, from a proportional-odds model (_proportional_odds);
    whether an ordered option at all, against each other option ("Not answered",
    "never asked"), from a multinomial logistic model. `ordered`: the indices of the
    ordered options, in order (_ordinal). `strength`: as C in LogisticRegression."""
    ordered = np.asarray(ordered, dtype=int)
    m = len(ordered)
    rank = np.full(k, -1)
    rank[ordered] = np.arange(m)
    others = np.array([j for j in range(k) if rank[j] < 0], dtype=int)
    y = code[train]
    is_ordered = rank[y] >= 0
    if len(others):
        group = np.where(is_ordered, len(others), np.searchsorted(others, y))
        if len(np.unique(group)) < 2:
            share = np.bincount(group, minlength=len(others) + 1) + PRIOR
            top = np.tile(share / share.sum(), (len(test), 1))
        else:
            model = LogisticRegression(C=strength, max_iter=300)
            model.fit(features[train], group)
            top = np.zeros((len(test), len(others) + 1))
            top[:, model.classes_] = model.predict_proba(features[test])
    else:
        top = np.ones((len(test), 1))
    rows = train[is_ordered]
    level = rank[code[rows]]
    if len(rows) == 0 or len(np.unique(level)) < 2:
        within = np.tile(pooled[ordered] / pooled[ordered].sum(), (len(test), 1))
    else:
        within = _proportional_odds(features[rows], level, m, strength, features[test])
    p = np.zeros((len(test), k))
    p[:, ordered] = top[:, -1:] * within
    if len(others):
        p[:, others] = top[:, :-1]
    return p


def _proportional_odds(x, y, m, strength, x_new):
    """A cumulative logit model, P(level <= j) = expit(cut_j - x . slope), fit by
    L-BFGS on the mean log loss plus |slope|^2 / (2 * strength * rows), which is
    LogisticRegression's penalty for C = strength. The cutpoints are kept in order by
    fitting the first one and the logs of the gaps. Returns the level probabilities
    for x_new."""
    x = x.astype(np.float64)
    n, d = x.shape
    penalty = 1.0 / (strength * n)
    upper = np.minimum(y, m - 2)
    lower = np.maximum(y - 1, 0)
    has_upper, has_lower = y < m - 1, y > 0
    counts = np.bincount(y, minlength=m) + PRIOR
    cumulative = np.cumsum(counts)[:-1] / counts.sum()
    start = np.log(cumulative / (1 - cumulative))
    theta = np.concatenate([[start[0]], np.log(np.maximum(np.diff(start), 1e-3)),
                            np.zeros(d)])

    def unpack(t):
        return np.cumsum(np.concatenate([[t[0]], np.exp(t[1:m - 1])])), t[m - 1:]

    def objective(t):
        cut, slope = unpack(t)
        eta = x @ slope
        su = np.where(has_upper, expit(cut[upper] - eta), 1.0)
        sl = np.where(has_lower, expit(cut[lower] - eta), 0.0)
        prob = np.maximum(su - sl, 1e-12)
        loss = -np.log(prob).mean() + 0.5 * penalty * slope @ slope
        du = np.where(has_upper, su * (1 - su), 0.0)
        dl = np.where(has_lower, sl * (1 - sl), 0.0)
        g_slope = x.T @ ((du - dl) / prob / n) + penalty * slope
        g_cut = (np.bincount(upper, weights=np.where(has_upper, -du / prob, 0.0),
                             minlength=m - 1)
                 + np.bincount(lower, weights=np.where(has_lower, dl / prob, 0.0),
                               minlength=m - 1)) / n
        tail = np.cumsum(g_cut[::-1])[::-1]
        return loss, np.concatenate([[g_cut.sum()], np.exp(t[1:m - 1]) * tail[1:], g_slope])

    result = minimize(objective, theta, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=300))
    cut, slope = unpack(result.x)
    eta = x_new.astype(np.float64) @ slope
    cdf = expit(cut[None, :] - eta[:, None])
    cdf = np.hstack([np.zeros((len(eta), 1)), cdf, np.ones((len(eta), 1))])
    return np.maximum(np.diff(cdf, axis=1), 0.0)


def _logit(code, train, test, features, k, c, pooled):
    if len(np.unique(code[train])) < 2:
        return np.tile(pooled, (len(test), 1))
    model = LogisticRegression(C=c, max_iter=300)
    model.fit(features[train], code[train])
    p = np.zeros((len(test), k))
    p[:, model.classes_] = model.predict_proba(features[test])
    return p


def _trees(code, train, test, features, categorical, k, iterations, pooled):
    """Gradient-boosted trees. With iterations=None, early stopping picks the number on an
    internal split of the training rows, and that number is returned for the refit.
    `categorical`: None, or a mask of the columns to split on sets of levels."""
    if len(np.unique(code[train])) < 2:
        return np.tile(pooled, (len(test), 1)), iterations
    try:
        if iterations is None:
            model = HistGradientBoostingClassifier(
                max_iter=TREE_ITERATIONS, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=10, random_state=SEED, categorical_features=categorical,
                **TREES)
        else:
            model = HistGradientBoostingClassifier(
                max_iter=max(int(iterations), 1), early_stopping=False,
                categorical_features=categorical, **TREES)
        model.fit(features[train], code[train])
    except Exception:      # e.g. a class too rare for the stratified early-stopping split
        model = HistGradientBoostingClassifier(max_iter=50, early_stopping=False,
                                               categorical_features=categorical, **TREES)
        model.fit(features[train], code[train])
    p = np.zeros((len(test), k))
    p[:, model.classes_] = model.predict_proba(features[test])
    return p, model.n_iter_


def _parents_first(items, names):
    """Order items so that a PREDICT parent comes before the items it gates."""
    pending, order = set(names), []
    while pending:
        ready = [name for name in names if name in pending and (
            not items[name].get("gate") or items[name]["gate"]["parent"] not in pending)]
        if not ready:                      # a cycle should not exist; keep going anyway
            ready = [name for name in names if name in pending]
        for name in ready:
            pending.discard(name)
            order.append(name)
    return order


def _gate_key(record):
    """(parent, answers that open the gate) for a gated item, else None."""
    gate = record.get("gate")
    if not gate:
        return None
    return gate["parent"], tuple(sorted(map(str, gate["observed_if"])))


def _sibling_groups(items, names):
    """{gate key: the items in `names` that share it}, for keys held by two or more."""
    groups = {}
    for name in names:
        key = _gate_key(items[name])
        if key is not None:
            groups.setdefault(key, []).append(name)
    return {key: members for key, members in groups.items() if len(members) >= 2}


def _held_truth(items, name, gated_value, values, position, held_rows, k):
    """Each held-out row's level of `name` (never asked last), -1 if not one."""
    index = {option: j for j, option in enumerate(items[name]["values"])}
    answers = values[held_rows, position[name]]
    return np.where(_gated(answers, gated_value), k - 1,
                    np.array([index.get(answer, -1) for answer in answers]))


def _group_parent(items, names, gated_value, values, position, fit, held_rows, model,
                  final, final_held, cache):
    """One gate parent for siblings (v13): None for the named parent, or an alternative
    any of them proposes, whichever gives the lowest held-out log loss summed over the
    siblings (each at its own best direct-model share and temperature); an alternative
    must win by ALT_HELD nats per pooled held-out row. FREE when there is nothing to
    compare (each item then chooses alone, as in v12)."""
    candidates = []
    for name in names:
        try:
            for parent in _alternatives(items, name, gated_value, values, position, fit,
                                        final, final_held, cache):
                if parent not in candidates:
                    candidates.append(parent)
        except Exception:
            continue
    if not candidates:
        return None
    options = [None] + candidates
    totals = dict.fromkeys(options, 0.0)
    pooled = 0
    for name in names:
        (content, content_held), direct = model[name]
        if content_held is None:
            continue
        scores = {}
        for parent in options:
            try:
                held = _chain(items, name, gated_value, values, position, fit, held_rows,
                              content_held, final_held, parent=parent, cache=cache)
            except Exception:
                break
            k = held.shape[1]
            truth = _held_truth(items, name, gated_value, values, position, held_rows, k)
            usable = truth >= 0
            if usable.sum() < MIN_VALIDATION:
                break
            direct_held = direct[1] if direct is not None else None
            best = np.inf
            for s in (SHARES if direct is not None else (0.0,)):
                mixed = held if direct is None else s * direct_held + (1 - s) * held
                for t in TAUS:
                    best = min(best, _score(_sharpen(mixed, t)[usable], truth[usable], k))
            scores[parent] = best * usable.sum()
        else:
            for parent in options:
                totals[parent] += scores[parent]
            pooled += int(usable.sum())
    if pooled == 0:
        return FREE
    pick = min(options, key=lambda parent: totals[parent])
    if pick is not None and totals[pick] > totals[None] - ALT_HELD * pooled:
        pick = None
    return pick


def _final(items, name, gated_value, values, position, visible, fit, held_rows, targets,
           model, final, final_held, cache=None, parent_choice=FREE):
    """The item's distribution for the target rows and for the held-out rows (None when
    the held-out version could not be built; its children then keep their chains).

    Ungated: the modeled answer. Gated: a gate chain and the direct model, mixed in the
    share, then sharpened by the temperature, that score best on the held-out rows (all
    of them, "never asked" included); without a direct model, the chain with its
    temperature. The chain is the named parent's or, if its mixture scores better, an
    alternative parent's (_alternatives, v9). The held-out version of a chain is built
    from the parent's held-out prediction and from rates counted on the fit rows only."""
    (content, content_held), direct = model
    if not items[name].get("gate"):
        return content, content_held
    cache = {} if cache is None else cache
    try:
        chain = _chain(items, name, gated_value, values, position, visible, targets,
                       content, final, cache=cache)
    except Exception:
        chain = _with_rate(name, gated_value, values, position, visible, content)
    try:
        chain_held = _chain(items, name, gated_value, values, position, fit, held_rows,
                            content_held, final_held, cache=cache)
    except Exception:                      # e.g. the parent has no held-out prediction
        return chain, None
    try:
        k = chain.shape[1]
        index = {option: j for j, option in enumerate(items[name]["values"])}
        answers = values[held_rows, position[name]]
        truth = np.where(_gated(answers, gated_value), k - 1,
                         np.array([index.get(answer, -1) for answer in answers]))
        usable = truth >= 0
        if usable.sum() < MIN_VALIDATION:
            return chain, chain_held
        chains = [(None, chain_held)]
        if parent_choice is not FREE:       # v13: the siblings chose together
            if parent_choice is not None:
                try:
                    chains.append((parent_choice, _chain(
                        items, name, gated_value, values, position, fit, held_rows,
                        content_held, final_held, parent=parent_choice, cache=cache)))
                except Exception:
                    pass
        else:
            try:
                for parent in _alternatives(items, name, gated_value, values, position,
                                            fit, final, final_held, cache):
                    try:
                        chains.append((parent, _chain(
                            items, name, gated_value, values, position, fit, held_rows,
                            content_held, final_held, parent=parent, cache=cache)))
                    except Exception:
                        continue
            except Exception:
                chains = chains[:1]
        direct_target, direct_held = direct if direct is not None else (None, None)
        best, pick, share, tau = np.inf, 0, 0.0, 1.0
        named = None
        searched = (range(len(chains)) if parent_choice is FREE
                    else [len(chains) - 1])  # v13: only the siblings' parent
        for j in searched:
            held = chains[j][1]
            for s in (SHARES if direct is not None else (0.0,)):
                mixed = held if direct is None else s * direct_held + (1 - s) * held
                for t in TAUS:
                    score = _score(_sharpen(mixed, t)[usable], truth[usable], k)
                    if score < best:
                        best, pick, share, tau = score, j, s, t
            if j == 0:
                named = (best, share, tau)
        if (parent_choice is FREE and pick
                and best > named[0] - ALT_HELD):  # v10: an alternative must win clearly
            pick, (best, share, tau) = 0, named
        parent, chain_held = chains[pick]
        if parent is not None:
            try:
                chain = _chain(items, name, gated_value, values, position, visible,
                               targets, content, final, parent=parent, cache=cache)
            except Exception:              # keep the named parent's chain and its choice
                return _final_mix(direct, chain, chains[0][1], usable, truth, k)
        if direct is None:
            return _sharpen(chain, tau), _sharpen(chain_held, tau)
        return (_sharpen(share * direct_target + (1 - share) * chain, tau),
                _sharpen(share * direct_held + (1 - share) * chain_held, tau))
    except Exception:
        return chain, chain_held


def _final_mix(direct, chain, chain_held, usable, truth, k):
    """v8's choice: the named parent's chain mixed with the direct model, the share and
    the temperature chosen on the held-out rows."""
    direct_target, direct_held = direct if direct is not None else (chain, chain_held)
    best, share, tau = np.inf, 0.0, 1.0
    for s in (SHARES if direct is not None else (0.0,)):
        mixed = s * direct_held + (1 - s) * chain_held
        for t in TAUS:
            score = _score(_sharpen(mixed, t)[usable], truth[usable], k)
            if score < best:
                best, share, tau = score, s, t
    return (_sharpen(share * direct_target + (1 - share) * chain, tau),
            _sharpen(share * direct_held + (1 - share) * chain_held, tau))


def _levels(items, name, gated_value):
    """An item's options, with "never asked" last if it is gated: the columns of its
    final distribution."""
    return list(items[name]["values"]) + ([gated_value] if items[name].get("gate") else [])


def _level_codes(items, name, gated_value, values, position, cache):
    """Each row's answer to `name` as an index into _levels (-1 for anything else),
    computed once per predict() call."""
    key = ("codes", name)
    if key not in cache:
        index = {level: j for j, level in enumerate(_levels(items, name, gated_value))}
        cache[key] = np.array([index.get(answer, -1) if isinstance(answer, str) else -1
                               for answer in values[:, position[name]]])
    return cache[key]


def _asked(name, gated_value, values, position, cache):
    """True where the item holds a value (it was asked)."""
    key = ("asked", name)
    if key not in cache:
        cache[key] = ~_gated(values[:, position[name]], gated_value)
    return cache[key]


def _rates(code, asked, known, base):
    """P(asked | the parent's answer is level j) for every level: the share asked among
    the `known` rows with that answer, shrunk toward base[j] by GATE_PRIOR rows."""
    rows = known & (code >= 0)
    n = np.bincount(code[rows], minlength=len(base)).astype(float)
    m = np.bincount(code[rows], weights=asked[rows].astype(float), minlength=len(base))
    return (m + GATE_PRIOR * base) / (n + GATE_PRIOR)


def _chain(items, name, gated_value, values, position, known, rows, content, final,
           parent=None, cache=None):
    """The gate chain for `rows`: the answer among those asked times the chance of being
    asked, plus the "never asked" slot. The chance is P(asked | the parent's answer)
    averaged over the parent's predicted answers (a PREDICT parent) or read at its
    answer (a GIVEN parent); each answer's rate is counted on the `known` rows (v9,
    _rates). For the named parent the rates are shrunk toward v8's two: the chance of a
    skip where the gate opens and of an answer where it does not. For an alternative
    `parent` (an earlier PREDICT item) they are shrunk toward the item's overall rate."""
    cache = {} if cache is None else cache
    asked = _asked(name, gated_value, values, position, cache)
    stated = parent is None
    if stated:
        gate = items[name]["gate"]
        parent = gate["parent"]
        if parent not in final and parent not in position:
            return _with_rate(name, gated_value, values, position, known, content)
        opened = set(gate["observed_if"])
        parent_open = np.array([answer in opened for answer in values[:, position[parent]]])
        open_rows, closed_rows = known & parent_open, known & ~parent_open
        leak_open = ((~asked[open_rows]).sum() + PRIOR) / (open_rows.sum() + 2 * PRIOR)
        leak_closed = (asked[closed_rows].sum() + PRIOR) / (closed_rows.sum() + 2 * PRIOR)
    levels = (_levels(items, parent, gated_value) if parent in final
              else list(items[parent]["values"]))
    if stated:
        base = np.array([1 - leak_open if level in opened else leak_closed
                         for level in levels])
    else:
        base = np.full(len(levels), (asked[known].sum() + PRIOR) / (known.sum() + 2 * PRIOR))
    if parent in final:
        code = _level_codes(items, parent, gated_value, values, position, cache)
        rates = _rates(code, asked, known, base)
        if final[parent].shape[1] != len(levels):
            raise ValueError("parent distribution does not match its levels")
        p_asked = final[parent] @ rates
    else:                                  # a GIVEN parent: its answer is known
        index = {level: j for j, level in enumerate(levels)}
        code = np.array([index.get(answer, -1) for answer in values[:, position[parent]]])
        rates = _rates(code, asked, known, base)
        fallback = np.array([1 - leak_open if answer in opened else leak_closed
                             for answer in values[rows, position[parent]]])
        p_asked = np.where(code[rows] >= 0, rates[np.maximum(code[rows], 0)], fallback)
    p_asked = np.clip(p_asked, 0.0, 1.0)
    return np.hstack([p_asked[:, None] * content, (1 - p_asked)[:, None]])


def _alternatives(items, name, gated_value, values, position, fit, final, final_held,
                  cache):
    """Up to ALT_PARENTS earlier PREDICT items whose answers on the fit rows say who was
    asked `name` better, by ALT_MARGIN nats per row, than its named parent does, best
    first (v9). Each is scored by the log loss of its per-answer rates (_rates, shrunk
    toward the overall rate), counted and scored on the same rows. Left out: the named
    parent, items without a held-out prediction, and items gated by the same parent on
    the same answers (they only repeat this item's own gate)."""
    gate = items[name]["gate"]
    named = gate["parent"]
    if named not in position:
        return []
    asked = _asked(name, gated_value, values, position, cache)
    rows = fit
    total = int(rows.sum())
    if total == 0:
        return []
    overall = (asked[rows].sum() + PRIOR) / (total + 2 * PRIOR)

    def loss(code, base):
        r = np.clip(_rates(code, asked, rows, base), 1e-6, 1 - 1e-6)
        seen = rows & (code >= 0)
        n = np.bincount(code[seen], minlength=len(base)).astype(float)
        m = np.bincount(code[seen], weights=asked[seen].astype(float), minlength=len(base))
        unseen = rows & (code < 0)
        rest = unseen.sum()
        rest_m = asked[unseen].sum()
        o = np.clip(overall, 1e-6, 1 - 1e-6)
        return -(np.sum(m * np.log(r) + (n - m) * np.log(1 - r))
                 + rest_m * np.log(o) + (rest - rest_m) * np.log(1 - o)) / total

    opened = set(gate["observed_if"])
    if named in final:
        named_levels = _levels(items, named, gated_value)
        named_code = _level_codes(items, named, gated_value, values, position, cache)
    else:
        named_levels = list(items[named]["values"])
        index = {level: j for j, level in enumerate(named_levels)}
        named_code = np.array([index.get(answer, -1) for answer in values[:, position[named]]])
    parent_open = np.array([answer in opened for answer in values[:, position[named]]])
    open_rows, closed_rows = rows & parent_open, rows & ~parent_open
    leak_open = ((~asked[open_rows]).sum() + PRIOR) / (open_rows.sum() + 2 * PRIOR)
    leak_closed = (asked[closed_rows].sum() + PRIOR) / (closed_rows.sum() + 2 * PRIOR)
    named_base = np.array([1 - leak_open if level in opened else leak_closed
                           for level in named_levels])
    bar = loss(named_code, named_base) - ALT_MARGIN
    if bar <= 0:
        return []
    scored = []
    for other in final:
        record = items[other]
        if other in (name, named) or record["class"] != "PREDICT":
            continue
        if final_held.get(other) is None:
            continue
        other_gate = record.get("gate")
        if (other_gate and other_gate["parent"] == named
                and set(other_gate["observed_if"]) == opened):
            continue
        levels = _levels(items, other, gated_value)
        code = _level_codes(items, other, gated_value, values, position, cache)
        value = loss(code, np.full(len(levels), overall))
        if value < bar:
            scored.append((value, other))
    scored.sort(key=lambda pair: pair[0])
    return [other for _, other in scored[:ALT_PARENTS]]


def _with_rate(name, gated_value, values, position, known, content):
    """The "never asked" slot at the rate observed on the `known` rows."""
    child_na = _gated(values[:, position[name]], gated_value)
    rate = (child_na[known].sum() + PRIOR) / (known.sum() + 2 * PRIOR)
    return np.hstack([(1 - rate) * content, np.full((len(content), 1), rate)])


def _valid(p):
    """Rows that are finite, non-negative and sum to 1 (uniform where a row has no mass),
    so the output can't fail the grader's checks."""
    p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    total = p.sum(axis=1, keepdims=True)
    p = np.where(total > 0, p, 1.0)
    return p / p.sum(axis=1, keepdims=True)
