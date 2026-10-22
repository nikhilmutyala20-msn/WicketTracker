# WicketTracker — full-match wicket delivery clips

Reads the scoreboard of a full cricket match video with OCR, finds every wicket, and cuts one MP4 clip of the
**delivery that took the wicket** (run-up, delivery, dismissal, short aftermath).

Runs on Colab GPU or the Colab TPU runtime (EasyOCR itself is not TPU-native: the TPU runtime is initialised for
TPU-compatible stages and OCR runs on CPU/GPU).

## Run

Colab: open `notebooks/WicketTracker_FullMatch_Colab.ipynb` (GPU) or `notebooks/WicketTracker_FullMatch_Colab_TPU.ipynb`
(TPU), point it at your video, run all cells. Always do the 5-minute smoke test first.

CLI:

```bash
pip install -r requirements.txt
python main.py --video match.mp4 --output-dir output          # add --cpu if there is no GPU
python main.py --video match.mp4 --output-dir output --start 0 --end 300 --no-clips   # smoke test
python tools/audit_clips.py output                              # contact sheet + verdict per clip
python -m pytest -q                                             # unit/integration tests
```

## Output

```
output/
├── clips/                    one MP4 per CONFIRMED wicket: wicket_<id>_inn<n>_<over>_<ball>.mp4
│   └── candidates/           wickets the scoreboard shows but whose delivery/label is uncertain (review these)
├── run_report.json           START HERE: trust OK/REVIEW, wickets found vs. what the scoreboard reached, clip statuses
├── clip_manifest.json        per wicket: anchor_reason, validation (VERIFIED / UNRESOLVED / REVIEW_REQUIRED) and the reasons
├── review_queue.json         wickets with NO clip (unreliable scoreboard evidence): T, L, R, search window, reasons
├── events.json               every wicket event with its evidence
├── rejected_events.json      duplicates, DRS-overturned wickets, wickets the scoreboard took back
├── match_timeline.json, innings.json, chronology.json, drs.json, replay_segments.json
└── scoreboard_observations.json, tracker.json, runtime.json, checkpoint.json
```

`run_report.json` compares the wickets with a confirmed event against the highest wicket count the scoreboard reached in
each innings. `trust: REVIEW` lists exactly what to look at (missing wicket number, duplicate, unresolved clip,
scoreboard rarely readable). It cannot prove a clip is right — watch the clips or run `tools/audit_clips.py`.

## How a clip is placed

1. A wicket event exists only when the scoreboard wicket count goes up by one (or a dismissal graphic appears after the
   last readable scoreboard of an innings). A wicket the scoreboard takes back (DRS) is retracted.
2. The scoreboard lags the ball by ~7 s (caught) to 15–60 s (DRS, stumping, run-out), so the wicket-count change is only
   a **signal `T`**, never the delivery time. The delivery is bounded on the scoreboard timeline
   (`backend/full_match/wicket_localizer.py`):
   * `L` = change time of the previous confirmed delivery (when over.ball changes together with the wicket, the ball-change
     immediately before it);
   * `R` = first strong replay start at/after `T − 2 s`; if no replay is detected, `T + 12 s`;
   * the wicket delivery is the **last reliable live-wide (bowler-end) shot inside `[L, R]`**
     (`backend/common/delivery_shot.py` finds shots; replays, DRS/HawkEye overlays, slow motion, close-ups are rejected);
   * clip = `live_wide.start − 1 s` (never before `L`) to `live_wide.end + 7 s` (never after `R`).
   * **First ball of an innings** has no previous delivery: `L` is the innings' first reliable scoreboard appearance and the
     **first** live-wide shot in `[L, R]` is used.
3. **No guessing.** If the scoreboard / over.ball evidence is unreliable or missing (skipped balls, more than one wicket at
   once, regressed over.ball, low scoreboard confidence, previous ball-change too close to the wicket signal, no scoreboard
   transition at all such as an OUT-text-only wicket) or no reliable live-wide shot exists in `[L, R]`, **no clip is cut**.
   The wicket is listed in `review_queue.json` / `clip_manifest.json` (`REVIEW_REQUIRED`) with its search window and
   reasons, and `run_report.json` shows `trust: REVIEW`.
4. The cut clip is re-scanned: it must contain a live bowling shot and must not start inside it (unless it starts at `L`
   by design), otherwise it is marked `UNRESOLVED` with the reason (never silently `VERIFIED`).

Candidates (`clips/candidates/`) are cut only for wickets proven by a scoreboard count change. A wicket the scoreboard
keeps for `wicket_persist_seconds` (90 s) is promoted to CONFIRMED even if the ball number jumped or the DRS reader was
confused. OUT text alone never gets a clip.

Sponsor / LED-hoarding text that flashes "OUT" is **learned from the video** (the same text in 3 separate episodes
≥ 120 s apart) — there is no per-broadcaster list. Optional extras: `Config.ad_board_markers`.

## Settings

All in `backend/common/config.py`, each documented inline. CLI switches: `--no-shot-anchor` (legacy scoreboard anchor),
`--no-candidate-clips`, `--sample-seconds`, `--checkpoint-seconds`, `--start/--end`.

## Known limits (be honest with yourself)

* Verified by unit/integration tests and by watching clips from one India v West Indies T20 broadcast. Picture
  thresholds in `ShotParams` were calibrated there; another broadcaster may need retuning. Add real frames to
  `tests/data/frames` and a test in `tests/test_v10_real_frames.py`.
* A scoreboard that is never readable cannot give wickets; `run_report.json` says so.
* A wicket whose over.ball is ambiguous keeps `delivery: null` (label `x`) and gets **no clip** — it goes to `review_queue.json`.
* With no replay detected, `R = T + 12 s` can include post-wicket wide shots (e.g. batter walking off) if the replay was
  missed; the clip is then the last live-wide shot before `R`. Tunables: `wicket_*` in `backend/common/config.py`.
* No OCR pipeline is perfect: run, read `run_report.json`, audit the clips.

## Layout

`backend/common` scoreboard parsing, OCR, replay/DRS detection, delivery-shot search, clip extraction ·
`backend/full_match` chronology, wicket detection, the pipeline · `backend/tpu` TPU runtime ·
`tools` audit_clips, run_clips (Tesseract stand-in for OCR-less environments) · `tests` + `tests/data/frames` real frames.
\n\n## OCR / wicket verification safety fix\n\nThe innings-start backfill now emits `CANDIDATE` events rather than `CONFIRMED` events when the first readable scoreboard already shows wickets down. This prevents the system from claiming a wicket delivery/time without an observed scoreboard transition or delivery evidence. The scoreboard count still remains evidence that one or more wickets fell before the scoreboard became readable.\n