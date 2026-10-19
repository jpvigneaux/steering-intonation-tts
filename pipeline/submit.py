#!/usr/bin/env python3
"""Submit a job to the cluster, with every site-specific value read from the configuration.

The batch payloads under ``slurm/`` carry no scheduler directives and no absolute paths, so
nothing in the repository names an account, a partition, an interpreter or a cache location.
Those come from the ``slurm`` section of your configuration file and are passed to ``sbatch``
here.

Resources are declared per job below and are sized from what each job actually does. After a
run, check the true peak with ``seff <JOBID>`` and tighten them: an over-large memory request
inflates fairshare billing, and an over-long walltime blocks backfill so the job waits behind
work it could have slotted between.

Prints the command and submits nothing unless the submission is confirmed with ``--submit``.

Examples::

    python pipeline/submit.py interventions                      # print, submit nothing
    python pipeline/submit.py --submit interventions
    python pipeline/submit.py --submit interventions --asr-only  # retry stage 2 alone

Everything after the job name is handed to the payload untouched, so this script's own
options must come before it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config  # noqa: E402

#: name -> (payload, resources). Sized from the work, not copied from a training script.
JOBS = {
    "tests": {
        "script": "pipeline/slurm/run_tests.sh",
        "gpu": False,
        # Sized from seff (job 7323229, 342 tests): 78 s wall, 724 MB, 1 core at 23%.
        # Memory 724 MB x 1.3 = 941 MB -> 950M; time 78 s x 1.4 = 109 s -> 3 minutes.
        "time": "00:03:00",
        "mem": "950M",
        "cpus": 1,
    },
    "interventions": {
        "script": "pipeline/slurm/run_interventions.sh",
        "gpu": True,
        # Sized from seff: decode+ASR of 432 recordings ran 2m29s using 1.9 GB.
        "time": "00:15:00",     # two model loads, ~450 decodes, ~450 transcriptions
        "mem": "4G",            # two small models, one utterance at a time, tensors on device
        "cpus": 2,              # audio I/O only; CPU utilization was ~0.9 of one core
    },
    "interventions-s2": {
        # The same payload with SYSTEM=s2 set by the caller: S2's codec decodes the swaps, so it
        # runs in the S2 environment. No decode-only S2 job has been measured; the default
        # resources are the CSM entry's, and callers pass --time/--mem derived for their run.
        "script": "pipeline/slurm/run_interventions.sh",
        "gpu": True,
        "time": "00:15:00",
        "mem": "4G",
        "cpus": 2,
        "python_key": "fish_python",
    },
    "analysis": {
        "script": "pipeline/slurm/run_analysis.sh",
        "gpu": False,
        # Sized from seff: 25s wall, 239 MB. Kept well above that so a larger generation
        # set does not need a resubmission to finish.
        "time": "00:15:00",     # F0 over ~430 short recordings, then a 10k paired bootstrap
        "mem": "2G",
        "cpus": 2,
    },
    "fish-pilot": {
        "script": "pipeline/slurm/run_fish_s2_pilot.sh",
        "gpu": True,
        # A 4B model in bf16 plus a codec, and twelve short generations. Unmeasured: this is
        # the first S2 run, so the request is a guess to be tightened from seff afterwards.
        "time": "00:40:00",
        "mem": "32G",
        "cpus": 4,
        "python_key": "fish_python",
    },
    "fish-generate": {
        "script": "pipeline/slurm/run_fish_s2_generate.sh",
        "gpu": True,
        # Sized from the pilot (job 5365458): 18.8 GB, and twelve generations in 3m21s
        # including model load. 36 recordings is a few minutes more. Memory 18.87 x1.15 = 22G,
        # the measured figure of every S2 job since; one core, at 67% of it.
        "time": "00:30:00",
        "mem": "22G",
        "cpus": 1,
        "python_key": "fish_python",
    },
    "generate": {
        "script": "pipeline/slurm/run_generate.sh",
        "gpu": True,
        # Sized from seff of the job that wrote the 156-recording set (5438751): 9m32s wall,
        # 12.23 GB Utilized, 4 cores at 23% (0.9 of one). Memory 12.23 x 1.15 = 14G; one core.
        # Time: that set's 2.7 s per recording does NOT carry over to a human-reference context
        # with --deterministic, measured at 5-6 s per recording (jobs 6597727/29, 18 Sept), after
        # a ~2.5 min model load. One voice of the multi-voice design is 36 recordings:
        # 2.5 + 36 x 5.5 s = 5.8 min, so 8. Pass --time for another size, rescaled the same way.
        "time": "00:08:00",
        "mem": "14G",
        "cpus": 1,
    },
    "steer-loop": {
        "script": "pipeline/slurm/run_steer_loop.sh",
        "gpu": True,
        # In-loop steering regenerates from the replayed prefix, so it costs a generation per
        # run. From seff of the worst Family D CSM generation (7022115): 18 recordings in 4:03,
        # 12.24 GB, 1 core at 81%; 7.6 s a recording after a 39 s warm-up, ~75 s of loading.
        # Default: the pilot's 48 runs (12 targets x 4), 75 + 39 + 48 x 7.6 = 479 s, x1.4 = 11.2
        # -> 12 min; 12.24 x1.15 = 14.1 -> 14G. Pass --time rescaled to the run count otherwise.
        # Margin: x1.15, not x1.3. This job type has been measured dozens of times and
        # sits within 0.1 GB of its figure every run; the partition bills memory at 1.0
        # per GB against 4.5 for a whole A100, so the margin is most of what the job
        # costs in fair share. Anything not yet measured keeps x1.3.
        "time": "00:12:00",
        "mem": "14G",
        "cpus": 1,
    },
    "steer-loop-s2": {
        "script": "pipeline/slurm/run_steer_loop.sh",
        "gpu": True,
        # From seff of the worst Family D S2 generation (7022112): 18 recordings in 4:57,
        # 18.86 GB, 1 core at 69%; 6.5 s a recording after ~181 s of loading and reference
        # encoding. Pilot default, 48 runs: 181 + 48 x 6.5 = 493 s, x1.4 = 11.5 -> 12 min;
        # 18.86 x1.15 = 21.7 -> 22G. Rescale --time to the run count otherwise.
        # Margin: x1.15, not x1.3. This job type has been measured dozens of times and
        # sits within 0.1 GB of its figure every run; the partition bills memory at 1.0
        # per GB against 4.5 for a whole A100, so the margin is most of what the job
        # costs in fair share. Anything not yet measured keeps x1.3.
        "time": "00:12:00",
        "mem": "22G",
        "cpus": 1,
        "python_key": "fish_python",
    },
    "align": {
        "script": "pipeline/slurm/run_align.sh",
        "gpu": False,
        # Sized from seff of the aligner's validation (job 6589439): two 156-recording
        # alignments in 4m07s, 2.15 GB Utilized, 2 cores at 36% (0.7 of one). Memory 2.15 x 1.3
        # = 2.8G; one core, so pass --jobs 1 to align.py. One voice set is 36 recordings:
        # about 2 min per 156 at two workers, so under a minute of aligning plus MFA's start-up;
        # 5 minutes. Rescale --time for a larger set.
        "time": "00:05:00",
        "mem": "2800M",
        "cpus": 1,
    },
    "joint-direction": {
        "script": "pipeline/slurm/run_joint_direction.sh",
        "gpu": False,
        # Sized from seff of cumulative-probe (5511309), which also loads every archive with its
        # backbone states: 6m46s over 156 archives, 1.14 GB Utilized, 2 cores at 85%. Memory
        # 1.14 x 1.3 = 1.5G; 2 cores. Three voices x 36 recordings = 108 archives: 6m46s x
        # 108/156 = 4m41s, an upper bound since one mean difference is lighter than its probes.
        "time": "00:05:00",
        "mem": "1500M",
        "cpus": 2,
    },
    "controls": {
        "script": "pipeline/slurm/run_controls.sh",
        "gpu": True,
        # CSM generation, ~144 recordings per design. Sized from the generation job's seff.
        "time": "01:00:00",
        "mem": "16G",
        "cpus": 4,
    },
    "backbone-patch": {
        "script": "pipeline/slurm/run_backbone_patch.sh",
        "gpu": True,
        # Fourteen variant directories, 2,772 recordings. Per recording: a depth re-run over
        # the window (E0 over the whole utterance), a Mimi decode and a Whisper pass. Sized
        # from seff (job 5453353): 57m27s and 11.98 GB. The memory request is raised well
        # above the observed peak rather than tightened to it -- that run finished with
        # essentially none to spare, and the peak is set by three models resident at once,
        # which does not shrink with a smaller sweep. Since 15 September the payload also runs
        # the verify_depth.py gate first (a generator load and three short greedy generations,
        # not yet measured by seff); check the next run's Elapsed before reusing this time.
        "time": "01:30:00",
        "mem": "20G",
        "cpus": 4,
    },
    "onset-dose": {
        "script": "pipeline/slurm/run_onset_dose.sh",
        "gpu": True,
        # Four level specifications, 154 recordings each: about 620 decodes plus the same
        # number of transcriptions. Sized from seff (job 5453369): 3m01s, 1.95 GB.
        "time": "00:10:00",
        "mem": "3G",
        "cpus": 2,
    },
    "backbone-steer": {
        "script": "pipeline/slurm/run_backbone_patch.sh",
        "gpu": True,
        # A follow-up steering sweep alone: seven coefficients, 1,078 recordings, and the
        # transcription stage restricted to the new variants by VARIANTS. Sized from the full
        # job's measured 146 ms per re-run frame.
        "time": "01:00:00",
        "mem": "20G",
        "cpus": 4,
    },
    "backbone-analysis": {
        "script": "pipeline/slurm/run_backbone_analysis.sh",
        "gpu": False,
        # F0 over ~3,400 decoded recordings, then a 10k paired bootstrap and a 10k cluster
        # bootstrap per read-out and direction. The analysis job's seff gives 25s and 239 MB
        # for ~430 recordings and one bootstrap; the cluster bootstrap is a Python loop and is
        # the slow part.
        "time": "01:00:00",
        "mem": "4G",
        "cpus": 2,
    },
    "backbone-patch-s2": {
        "script": "pipeline/slurm/run_backbone_patch_s2.sh",
        "gpu": True,
        # Derived from seff of the two nearest S2 jobs. Memory: every job that loads the 4B model
        # and the codec peaked at 18.8 GB (5365458, 5367869); x1.3 = 24.5 -> 25G. CPUs: those
        # jobs ran 4 cores at 20% efficiency, 0.83 of one core -> 1. Time: the S2 decode job
        # (5438710) decoded 180 recordings in 3m28s with a codec load per manifest, ~1.15 s a
        # recording; this run decodes 15 variants x 36 = 540, ~10.5 min, plus ~2 min for the
        # model load (pilot), ~2 min for the verification, and ~3 min for 576 transcriptions,
        # which have no S2 precedent and are the least certain term. ~18 min estimated.
        "time": "00:25:00",
        "mem": "25G",
        "cpus": 1,
        "python_key": "fish_python",
    },
    "style-tags": {
        "script": "pipeline/slurm/run_style_tags.sh",
        "gpu": True,
        # From seff of the published style-tag run (5508257): 228 recordings in 30:34, 18.86 GB,
        # 4 cores at 25% (one core's worth); ~7.4 s a recording after ~150 s of loading. The
        # default is one plan-prompted voice, 171 recordings: 150 + 171 x 7.4 = 1,415 s, x1.4 =
        # 33 -> 35 min. 18.86 x1.15 = 21.7 -> 22G, the S2 margin. Rescale --time for 228.
        "time": "00:35:00",
        "mem": "22G",
        "cpus": 1,
        "python_key": "fish_python",
    },
    "style-tags-contours": {
        "script": "pipeline/slurm/run_style_tags_contours.sh",
        "gpu": False,
        # From seff 5509602: 228 recordings in 73 s, 394 MB, 0.45 of a core. 73 x1.4 = 102 s
        # -> 3 min; 394 x1.3 = 512 -> 520M.
        "time": "00:03:00",
        "mem": "520M",
        "cpus": 1,
        "python_key": "fish_python",
    },
    "style-tags-asr": {
        "script": "pipeline/slurm/run_style_tags_asr.sh",
        "gpu": False,
        # Whisper-small on CPU (pass --device cpu). From seff 5509687: 228 recordings in 7:07,
        # 1.87 GB, 4 cores at 82% (3.3 in use). 427 s x1.4 = 598 s -> 10 min; 1.87 x1.3 = 2.4
        # -> 2500M; 4 cores.
        "time": "00:10:00",
        "mem": "2500M",
        "cpus": 4,
    },
    "cumulative-probe": {
        "script": "pipeline/slurm/run_cumulative_probe.sh",
        "gpu": False,
        # Reads ~156 archives, loads the Mimi codec on CPU for the codeword tables and the
        # quantizer's decode, then fits ~120 leave-one-sentence-out probes each with a 20-draw
        # permutation null. No audio, no encoder, no decoder. Sized from seff (job 5511309):
        # 6m46s and 1.14 GB, of which the codec is most of the memory.
        "time": "00:20:00",
        "mem": "3G",
        "cpus": 2,
    },
}


def build_command(name: str, payload_args: list[str], overrides: dict | None = None,
                  after: str | None = None) -> list[str]:
    job = JOBS[name]
    slurm = config.get("slurm")
    job = {**job, **{k: v for k, v in (overrides or {}).items() if v}}

    command = [
        "sbatch",
        f"--account={slurm['account']}",
        f"--partition={slurm['gpu_partition'] if job['gpu'] else slurm['cpu_partition']}",
        "--nodes=1", "--ntasks-per-node=1",
        f"--cpus-per-task={job['cpus']}",
        f"--time={job['time']}",
        f"--mem={job['mem']}",
        f"--job-name=avsi-{name}",
        f"--output={Path(config.get('logs_root'))}/avsi-{name}-%j.log",
    ]
    if job["gpu"] and slurm.get("gpu_gres"):
        command.append(f"--gres={slurm['gpu_gres']}")
    if after:
        # afterok, not afterany: a stage whose input failed its own check must not start. The
        # alignment job exits non-zero on any doubtful grid for exactly this reason.
        command.append(f"--dependency=afterok:{after.replace(',', ':')}")

    # A job may need an interpreter other than the default one: fish-speech pins a torch
    # version the rest of the pipeline does not use, so S2 lives in its own environment.
    interpreter = slurm[job.get("python_key", "python")]
    exports = [f"PYTHON={interpreter}", f"REPO={_ROOT}",
               "TOKENIZERS_PARALLELISM=false"]
    if slurm.get("hf_home"):
        exports += [f"HF_HOME={slurm['hf_home']}", "HF_HUB_OFFLINE=1"]
    # ALL keeps the submitting environment; --export=NONE would strip PATH and leave the job
    # unable to find bash.
    command.append("--export=ALL," + ",".join(exports))

    return command + [str(_ROOT / job["script"]), *payload_args]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--submit", action="store_true",
                        help="Actually queue it. Without this the command is only printed.")
    # A job's table entry is sized for one particular run. A run of a different size needs its
    # own numbers -- rescaled from seff, not the table's -- and overriding here is better than
    # editing the table, which would leave the wrong numbers behind for the next caller.
    parser.add_argument("--time", dest="time_override", default=None,
                        help="Override the walltime, e.g. 00:25:00. Rescale from seff Elapsed "
                             "by this run's item count; do not reuse another run's value.")
    parser.add_argument("--mem", dest="mem_override", default=None,
                        help="Override the memory, e.g. 24G. From seff 'Memory Utilized' x1.3.")
    parser.add_argument("--cpus", dest="cpus_override", type=int, default=None,
                        help="Override cpus-per-task. From seff Cores x CPU Efficiency.")
    parser.add_argument("--after", default=None,
                        help="Job id(s), comma-separated, that must finish successfully first.")
    # REMAINDER, not "*": the payload takes option-like arguments such as --asr-only, and a
    # plain positional would have argparse try to interpret them as its own options.
    parser.add_argument("payload_args", nargs=argparse.REMAINDER,
                        help="Arguments passed through to the payload script, e.g. --asr-only.")
    args = parser.parse_args()

    overrides = {"time": args.time_override, "mem": args.mem_override,
                 "cpus": args.cpus_override}
    command = build_command(args.job, args.payload_args, overrides, after=args.after)
    resources = {**JOBS[args.job], **{k: v for k, v in overrides.items() if v}}
    print(f"{args.job}: {resources['time']} walltime, {resources['mem']} memory, "
          f"{resources['cpus']} cpus, gpu={resources['gpu']}")
    print("\n  " + " \\\n    ".join(command) + "\n")

    if not args.submit:
        print("Not submitted. Re-run with --submit to queue it.")
        return
    if not shutil.which("sbatch"):
        raise SystemExit("sbatch not found; are you on a submit host?")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode:
        raise SystemExit(result.returncode)
    print("\nAfter it finishes, size the next run from the true peak:  seff <JOBID>")


if __name__ == "__main__":
    main()
