# Dracula in Barcelona — Technical Document

**More than a Hack 2027, Phase 1: Ball Possession** · BAR-ATH · output `Dracula in Barcelona.csv`

## 1. Approach

The pipeline has **two stages**. First it decides whether the ball is in play (*live* or *dead*).
Then, inside each live stretch, it decides whether Home or Away has the ball.

**Models.** All the probabilities come from models trained **only on Metrica's public Sample
Games 1–3** ([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)). The dataset comprises 3 complete matches (~386,000 tracking frames at 25 fps: Game 1 ~145k, Game 2 ~141k, Game 3 ~100k; expanded 8× across pitch symmetries to ~1.16M frame instances). Labels were derived mechanically from Metrica's timestamped event logs (passes, carries, shots, set pieces) without any manual annotation on the target match:

- Two gradient-boosting models and two sequence models (a dilated residual TCN, and a TCN with a
  BiGRU on top) estimate P(Home | live). The sequence models also estimate the probability that
  a hand-over happens at each frame (P(boundary)).
- A gradient-boosting model and a BiGRU estimate P(dead) from **player features only**.

**Decoding.** An exact 2-state Viterbi decoder turns each probability track into segments. It
charges a penalty for every switch, and the penalty is lowered where P(boundary) is high.
Structural rules then handle restarts, goals and goalkeeper holds.

**Why this design:**

- **No tuning on BAR-ATH.** Seventeen public possessions are too few to choose between models.
  The organisers show that even a single smoothing window chosen on them ranks the wrong
  baseline first. So all model weights and decoder parameters are fitted on Metrica, with
  leave-one-game-out cross-validation and then a Pareto search that treats each Metrica game's
  spellF1 as a separate objective (`pareto_metrica.py`). The public window is used only to check
  that the pipeline is correct.
- **Simulated BAR-ATH noise.** In BAR-ATH the ball is missing on about 41 % of graded frames, and
  Metrica's tracking is clean. We hide the ball in the training games using the visible and
  missing run lengths measured on BAR-ATH (this uses no labels). We also add player jitter,
  dropouts and homography-style wobble. Each game is copied 8 times across the four football
  symmetries, and we average over the same symmetries at inference.
- **A player-only dead-ball model.** Metrica blanks the ball during stoppages, while BAR-ATH often
  keeps tracking it out of bounds. Ball features would not carry over from one to the other.
- **Viterbi rather than a majority vote.** spellF1 rewards whole possessions, so the number of
  switches matters more than per-frame accuracy.
- **Visibility-dependent blend.** Gradient boosting is sharp when the ball is visible, and the
  sequence models carry context through occlusions. The weight on the sequence models is 0.65
  when the ball is visible and 0.45 when it is not.

On Metrica out-of-fold data: **spellF1 0.791, macroF1 0.912**. On the 5-minute public ground truth sanity check (BAR-ATH), the identical frozen pipeline achieves **spellF1 0.671, macroF1 0.908** without any tuning or threshold selection on BAR-ATH. We expect the full-match BAR-ATH score to reflect this domain noise.

## 2. Assets: used and dropped

| asset | how it is used |
|---|---|
| **Tactical ATD** | All features. Goalkeeper slots are detected automatically per half. |
| **Broadcast ATD** | We align it to the tactical clock (matching team centroids and the ball), then use it to fill gaps in the tactical ball track. Where the two cameras disagree, we keep the track that is physically plausible and near a player. |
| **Broadcast homography** | Only its `valid` flag, as a TV-cut/replay signal (weak evidence of a stoppage). |
| **Smart Tagging** | `SET PIECES` and `GOALS` give the stoppage structure. Phase codes (build-up, transitions, defending…) give a weak team prior and mark likely hand-overs. |
| **[Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data)** (Games 1–3, public open-source) | Training data (~386k frames at 25 fps; event logs converted mechanically to possession ground truth). |
| **Public 5-min ground truth** | Sanity checks only (achieves spellF1 0.671, macroF1 0.908). Never used for training or for choosing parameters. |
| ❌ **Videos, tactical homography** | Dropped. No computer vision; the ATD is already in pitch coordinates. |

**Also tried and dropped:**

- Touch-event features: noisy with the ball so often missing.
- Ball features in the dead-ball model: they do not transfer from Metrica to BAR-ATH.
- Focal loss: plain BCE keeps the log-odds calibrated for Viterbi.
- A fixed 50/50 model blend: replaced by the visibility-dependent blend above.

## 3. Possession rules

- **Controlled possession.** In the training labels, a spell **under 2 s with fewer than 2
  controlled events** (pass, carry, set piece, shot) does not count as a change of possession.
  This rules out single touches, deflections and aerial duels won without control.
- **Loose balls and challenges.** A switch costs 28 log-units, so a 50/50 duel keeps the ball
  with the last team in possession. Spells shorter than **2 s** are merged into a neighbour,
  unless they sit at the edge of a live segment and the model is very confident. A long spell is
  split if the frame-level model holds the other team for **3 s or more**, with a mean
  probability of at least 0.72. Each hand-over can then move by up to ±14 frames to fit the
  evidence.
- **Restarts.** In a set piece, the ball is dead until it first moves faster than 2 m/s. After a
  goal, it is dead from the goal until the kick-off leaves the centre circle. A ball out of
  bounds is strong evidence of dead. Stoppages last at least 20 frames, and both short live and
  short dead islands are cleaned up.
- **Goalkeeper holds.** A goalkeeper who catches the ball and then distributes it (ball in the
  team's own box, followed by a build-up or long-ball tag) counts as **live, owned by the
  goalkeeper's team**.
- **Twilight frames.** A short, low-confidence spell right next to a stoppage becomes `D`.
- **`X`** is used only outside the two halves. During play we always answer H, A or D.

## 4. Limitations and next steps

- **Domain gap.** Three Metrica games are the only training data. The simulated noise matches how
  often and how long BAR-ATH loses the ball, but not *when* it happens (crosses, crowded boxes).
- **Turnovers while the ball is missing** are often lost or detected late. *Next:* re-detect the
  ball from the tactical video.
- **Rule thresholds are set by hand.** The decoder parameters are calibrated, but the thresholds
  in the structural rules are fixed in code. *Next:* fold them into the Pareto search.
- **Match-specific code.** The halftime frame and the team names are hard-coded.
- **Smart Tagging is automatic**, so wrong tags lead to wrong priors.
- **Quick restarts** (fast throw-ins and free kicks) can be absorbed into live play.
- **Unpinned packages.** `numpy`, `pandas` and `scipy` are not pinned. Use Docker as the
  reference environment.

## 5. Reproduction

Unpack the data bundle into `./data/tactical` and `./data/broadcast`, then run:

```bash
./reproduce.sh            # Docker if available, otherwise an isolated venv (--local forces the venv)
```

The script parses the assets into `cache/`, runs deterministic CPU inference from the shipped
`models/`, decodes, writes `evaluation/results/Dracula in Barcelona.csv` and validates it.
To do the same by hand:

```bash
pip install -r requirements.txt -r requirements-starter.txt
python3 starter/prep.py --all
python3 starter/algorithmic_baseline.py
python3 evaluation/validate_submission.py "evaluation/results/Dracula in Barcelona.csv"
```

To retrain from scratch (optional; this clones the public repository `https://github.com/metrica-sports/sample-data`):

```bash
python3 starter/metrica_possession_model.py
python3 starter/pareto_metrica.py --save-model
```

No match data is committed: `data/`, `cache/` and submission CSVs are all gitignored.
