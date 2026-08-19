# Multi-seed sweep data

Raw per-checkpoint stress-test results backing the multi-seed tables in
[`../RESULTS.md`](../RESULTS.md#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).
Every checkpoint was evaluated on the same 500,000-path regime-switching
scenario at seed 42 used throughout that document, so these numbers are
directly comparable to its other tables.

Each entry carries the training args the checkpoint was saved with, so the
condition each row belongs to is recoverable from the data itself rather than
only from its filename.

| File | Contents |
|---|---|
| `RESULT_gru_wgan_5seed.json` | GRU (WGAN-GP), 5 seeds x {baseline, `--grad-clip-norm 1.0`} — the retracted fix |
| `RESULT_gru_tg_5seed.json` | GRU (TimeGAN), 5 seeds x {baseline, `--moneyness-clip -0.15 0.10`} — the retracted fix |
| `RESULT_rnn_tg_5seed.json` | Basic RNN (TimeGAN), 5 seeds x {baseline, `--lr 1e-3`, `--lr 1e-3` + clip} — the promoted fix |
| `RESULT_gru_clip_threshold.json` | GRU (WGAN-GP) baselines vs. `--grad-clip-norm` 0.05 (5 seeds) and 0.10 (3 seeds) |
| `RESULT_timegan_rows_5seed.json` | TimeGAN row re-anchoring: MLP (5 seeds), LSTM `--slow-ramp-fraction 0.05` vs. untreated LSTM (5 seeds each) |

The checkpoints themselves are gitignored (`checkpoints/`), so these summaries
are the durable record — regenerating them means retraining, which is the
situation that prompted this whole exercise.

All of it is produced by `../src/backtester/stress_eval.py`, which is now
committed for the same reason these records are: it previously lived only in
a scratch directory outside the repo and was lost with it.
