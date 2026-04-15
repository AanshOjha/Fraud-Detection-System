Right now, it is **not automatically time-triggered by the batch files themselves**.

At the moment the logic is only:
- all batch files exist in `data/raw/batches/`
- `main.py` reads however many batches you tell it to use
- more batches included = “more new data has arrived”

So the “next week’s data” idea is currently **simulated by batch order**, not by real calendar waiting.

**How it works now**
If you run:

```powershell
python main.py --max-batches 2
```

it treats:
- `batch_001.csv`
- `batch_002.csv`

as the only available historical data.

Then later if you run:

```powershell
python main.py --max-batches 3
```

it now treats:
- `batch_003.csv`

as the newly arrived batch.

So the “time progression” is manual right now.

**How to think about it**
The batches represent:
- `batch_001` = older data
- `batch_002` = later data
- `batch_003` = next period
- `batch_004` = next period after that

You are simulating time by revealing more batches over time.

**If you want true automatic behavior**
You need one extra rule for retraining trigger. Usually one of these:

1. **Manual trigger**
- you decide when to run `main.py --max-batches N`
- easiest for demo

2. **Scheduler trigger**
- every week, run training
- training checks if a new batch exists
- if yes, retrain and promote if better

3. **New-file trigger**
- if a new `batch_00x.csv` appears, retrain automatically
- closer to production behavior

**For your local project, best beginner approach**
Use a tiny state file like:
- last processed batch = `batch_002.csv`

Then next time pipeline runs:
- scan `data/raw/batches/`
- see `batch_003.csv` is new
- include it
- retrain
- update state to `batch_003.csv`

That is the cleanest “realistic” local MLOps behavior.

**So currently your repo is**
- good for simulated sequential retraining
- not yet automatically aware of “this is next week”

To make it behave like weekly arrival, we would add either:
- a scheduler, or
- a `last_processed_batch.json` state file

**Simple mental model**
Current version:
- “I choose how many weeks of data exist”

Better next version:
- “The pipeline remembers what it already used, and picks up only newly available batches”

If you want, I can implement that next in a very simple beginner-friendly way:
- add `pipeline_state.json`
- detect newly available batches
- retrain only when new batch files appear
- optionally add a local scheduled task script for weekly runs.