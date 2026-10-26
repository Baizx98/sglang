"""Continue a running first pilot with the expanded, resume-safe data queue."""
import argparse,json,subprocess,time
from pathlib import Path
from horizonkv_single_runtime import REPO,update,notify,save

p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prior-pid',type=int,required=True);a=p.parse_args();root=a.run_dir.resolve()
while True:
    status=Path(f'/proc/{a.prior_pid}/stat')
    if not status.exists() or status.read_text().split()[2]=='Z':break
    time.sleep(15)
state=json.loads((root/'state.json').read_text())
if state['status']=='failed':
    save(root/'extension-waiter.json',dict(status='not_started_due_to_pilot_failure',state=state));raise SystemExit(1)
update(root,status='running',phase='resuming_pilot_and_extension_queue',detail='First pilot finished; resume existing data and run authorized expansion')
cmd=[str(REPO/'.venv/bin/python'),'scripts/keye_trace/continue_horizonkv_fixed_budget.py','--run-dir',str(root)]
with (root/'expanded-continuation.log').open('a') as log:
    child=subprocess.Popen(cmd,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'expanded-queue-launch.json',dict(pid=child.pid,command=cmd));code=child.wait()
raise SystemExit(code)
