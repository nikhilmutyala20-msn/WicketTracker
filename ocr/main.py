import argparse, json, os
from backend.common.config import Config
from backend.full_match.pipeline import run_full_match


def main():
    p = argparse.ArgumentParser(description='WicketTracker Full Match OCR — Colab TPU-aware')
    p.add_argument('--video', required=True)
    p.add_argument('--output-dir', default='output')
    p.add_argument('--cpu', action='store_true', help='force CPU OCR (recommended on TPU runtimes)')
    p.add_argument('--require-gpu', action='store_true', help='legacy NVIDIA GPU requirement')
    p.add_argument('--disable-tpu', action='store_true', help='do not initialize TensorFlow TPU runtime')
    p.add_argument('--checkpoint-seconds', type=float, default=None)
    p.add_argument('--checkpoint-warmup', type=float, default=None)
    p.add_argument('--sample-seconds', type=float, default=1.0)
    p.add_argument('--no-clips', action='store_true')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float, default=None)
    p.add_argument('--preroll', type=float, default=None)
    p.add_argument('--postroll', type=float, default=None)
    p.add_argument('--global-width', type=int, default=None)
    p.add_argument('--trigger-interval', type=float, default=None)
    p.add_argument('--rediscover-every', type=float, default=None)
    p.add_argument('--no-shot-anchor', action='store_true', help='v10: disable picture-based delivery localisation (legacy anchor)')
    p.add_argument('--no-candidate-clips', action='store_true', help='v10: do not cut clips/candidates/ for unconfirmed wicket events')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args()
    cfg = Config()
    cfg.require_gpu = bool(a.require_gpu)
    cfg.enable_tpu_runtime = not a.disable_tpu
    if a.checkpoint_seconds is not None: cfg.checkpoint_seconds = max(10.0, a.checkpoint_seconds)
    if a.checkpoint_warmup is not None: cfg.checkpoint_warmup_seconds = max(0.0, a.checkpoint_warmup)
    if a.preroll is not None: cfg.preroll_seconds = a.preroll
    if a.postroll is not None: cfg.postroll_seconds = a.postroll
    if a.global_width is not None: cfg.global_scan_max_width = a.global_width
    if a.trigger_interval is not None: cfg.trigger_global_interval = a.trigger_interval
    if a.rediscover_every is not None: cfg.rediscover_every_seconds = a.rediscover_every
    if a.no_shot_anchor: cfg.use_shot_anchor = False
    if a.no_candidate_clips: cfg.candidate_clips = False
    events = run_full_match(a.video, a.output_dir, gpu=(not a.cpu and a.require_gpu),
                            sample_seconds=a.sample_seconds, extract_clips=not a.no_clips,
                            cfg=cfg, start_s=a.start, end_s=a.end)

    # Verification-only report: does not modify the existing wicket detection/localization logic.
    # It simply records the delivery/time/count information already produced by WicketEvent.
    def _fmt_ts(seconds):
        if seconds is None:
            return 'N/A'
        total = max(0, int(round(float(seconds))))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    report_path = os.path.join(a.output_dir, 'wicket_taking_deliveries.txt')
    confirmed = [e for e in events if e.status == 'CONFIRMED']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('WICKET-TAKING DELIVERY VERIFICATION REPORT\n')
        f.write('=' * 60 + '\n')
        f.write(f'Video: {a.video}\n')
        f.write(f'Total wicket events detected: {len(events)}\n')
        f.write(f'Confirmed wicket-taking deliveries: {len(confirmed)}\n')
        f.write('\n')

        for i, e in enumerate(confirmed, 1):
            f.write(f'WICKET #{i}\n')
            f.write(f'  Event ID          : {e.event_id}\n')
            f.write(f'  Innings           : {e.innings if e.innings is not None else "N/A"}\n')
            f.write(f'  Batting team      : {e.batting_team or "N/A"}\n')
            f.write(f'  Wicket count      : {e.wickets_after if e.wickets_after is not None else "N/A"}\n')
            f.write(f'  Delivery          : {e.delivery or "N/A"}\n')
            f.write(f'  Video time        : {_fmt_ts(e.timestamp)} ({e.timestamp:.3f}s)\n')
            f.write(f'  Delivery window   : {_fmt_ts(e.delivery_window_start)} -> {_fmt_ts(e.delivery_window_end)}\n')
            f.write(f'  Score before      : {e.runs_before if e.runs_before is not None else "N/A"}/{e.wickets_before if e.wickets_before is not None else "N/A"}\n')
            f.write(f'  Score after       : {e.runs_after if e.runs_after is not None else "N/A"}/{e.wickets_after if e.wickets_after is not None else "N/A"}\n')
            f.write(f'  Over/ball before  : {e.over_before if e.over_before is not None else "N/A"}.{e.ball_before if e.ball_before is not None else "N/A"}\n')
            f.write(f'  Over/ball after   : {e.over_after if e.over_after is not None else "N/A"}.{e.ball_after if e.ball_after is not None else "N/A"}\n')
            f.write(f'  Confidence        : {e.confidence:.3f}\n')
            f.write(f'  Status            : {e.status}\n')
            f.write(f'  DRS state         : {e.drs_state}\n')
            f.write('\n')

        non_confirmed = [e for e in events if e.status != 'CONFIRMED']
        if non_confirmed:
            f.write('NON-CONFIRMED EVENTS (for verification)\n')
            f.write('-' * 60 + '\n')
            for e in non_confirmed:
                f.write(f'Event {e.event_id}: status={e.status}, innings={e.innings}, '
                        f'wicket_after={e.wickets_after}, delivery={e.delivery or "N/A"}, '
                        f'time={_fmt_ts(e.timestamp)} ({e.timestamp:.3f}s), confidence={e.confidence:.3f}\n')

    if not a.quiet:
        print(json.dumps([e.to_dict() for e in events], indent=2, default=str))
    print(f'{len(events)} event(s); outputs in {a.output_dir}/')


if __name__ == '__main__':
    main()
