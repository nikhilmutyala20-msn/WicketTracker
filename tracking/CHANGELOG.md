# Changelog

## v12 (this version) -- wicket clip localisation on the scoreboard timeline
* New `backend/full_match/wicket_localizer.py`. T = wicket-count change (a signal, not the delivery time); L = previous
  delivery's ball-change; R = first strong replay start at/after T-2 s (else T+12 s). The clip is cut around the LAST reliable
  live-wide shot inside [L, R]: `start-1 s` (>= L) to `end+7 s` (<= R). First ball of an innings: L = first scoreboard
  appearance, FIRST live-wide shot.
* Unreliable / missing scoreboard evidence (skipped balls, multi-wicket jump, regressed over.ball, low confidence, previous
  ball-change < `min_ball_gap` before T, no scoreboard transition: back-fill / OUT-text-only wickets) or no live-wide shot in
  [L, R] -> NO clip. Wicket goes to `review_queue.json` + `clip_manifest.json` (`REVIEW_REQUIRED`); `run_report.json` -> REVIEW.
  The v10/v11 "wide fallback window" clip is gone (it was a guess).
* Fixes v11's `group`/first-shot pick: a stray earlier wide shot in the same 30 s group could be cut instead of the wicket ball
  (reproduced on a synthetic match: v11 cut [18 s, 40 s] and marked it VERIFIED; the wicket delivery was 40-44 s).
* Unchanged: OCR, DRS, tracking, YOLO, pitch detection, heatmaps, event detection / merge / `locate_delivery` evidence window.
* New config: `wicket_*` (replay lead/ahead/min length, no-replay R, live-wide min length, overlap tolerance, search cap,
  scoreboard confidence, clip pre/post/min).

## v11
* Candidate clips only for wickets proven by a scoreboard count change; OUT-text-only guesses never get a clip
  (2 of 5 V10 candidate clips were single runs).
* Scoreboard-proven wickets that the scoreboard keeps (90 s) are promoted from CANDIDATE to CONFIRMED when the only doubt was
  an ambiguous ball number or a confused DRS reading (e.g. an LBW overturned from NOT OUT to OUT).
* Sponsor / LED-board "OUT" text is learned from the video instead of a hard-coded list.
* Clips capped at 22 s (V10: 28 s) so they are not mostly celebration / the next ball.
* `run_report.json`: wickets found vs what the scoreboard reached, duplicates, unresolved clips, `trust: OK|REVIEW`.
* Package cleanup: legacy audits/notes, docs/, synthetic-only tools and cached files removed; one README.

## v10
Clip anchored on the live bowler-end shot found in the pictures (scoreboard lags the ball); DRS-reverted wicket retracted
and no false new innings; wickets already down when an innings is first read are back-filled; last-ball wickets promoted
from OUT text; DRS graphics / ad boards no longer count as dismissals; clips re-scanned after cutting.

## v9
Bowler-figures panel (`W-R (O.B)`, e.g. `2-26 (2.4)`) no longer mistaken for the scoreboard; `LAST WICKET` banner is not live
OUT evidence; ROI cache fix; CLAHE retry; OCR confusion table 1↔4, 4↔9; dense sampling when a read disagrees.

## v8.1
Clips are 10–15 s localised on a primary-play anchor instead of the wide delivery window.

## v7
Scoreboard fusion prefers the strongest cricket score in mixed OCR clusters; narrowly-gated stale-primary correction.

## v6 / TPU
Colab GPU + Drive workflow, Colab TPU runtime initialisation (`backend/tpu`), checkpointing.

## OCR reliability update (2026-10-16)
- Enhanced ROI OCR fallback now retries four complementary image variants: upscaled original, CLAHE contrast normalization, sharpened CLAHE, and adaptive thresholding.
- Overlapping OCR boxes across variants are deduplicated, retaining the highest-confidence reading so repeated passes do not artificially inflate candidate confidence.
- This fallback runs only after the tracked ROI has no acceptable scoreboard candidate, preserving the normal fast path.
- Validation performed on this update: 69 targeted parser/scoreboard/real-OCR/localization tests passed; the real EasyOCR test module was skipped when EasyOCR was not installed in the test environment. Full-match output still requires a fresh run and visual clip audit; OCR accuracy cannot be guaranteed without evaluating the actual source video.
