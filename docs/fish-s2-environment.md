# Building the Fish Audio S2 environment

S2 needs an environment of its own. `fish-speech` 2.0.0 pins `torch==2.8.0`, while the CSM half
of this repository runs on 2.4; installing S2 alongside it would upgrade torch underneath every
result the CSM pipeline has produced. The two environments never meet — `submit.py` lets a job
name its interpreter through `slurm.fish_python`.

Everything below runs on a login node. Compute nodes have no outbound network, so weights and
packages must be in place before a job starts.

## 1. Environment

```bash
mamba create -y -p <envs>/fish-s2-env python=3.11 pip
git clone --depth 1 https://github.com/fishaudio/fish-speech.git <code>/fish-speech
```

Put both under durable storage. An environment on an age-purged filesystem will disappear.

## 2. Dependencies, minus one

`pip install -e ".[cu128]"` **fails**: `pyaudio` needs portaudio system headers that are not
present, and pip rolls the whole transaction back, leaving nothing installed. `pyaudio` is used
for microphone I/O and nothing on the generation path touches it.

So install the dependency list without it, then the package itself with `--no-deps`:

```bash
python - <<'PY' > deps.txt
import tomllib, pathlib, re
data = tomllib.loads(pathlib.Path("fish-speech/pyproject.toml").read_text())
for d in data["project"]["dependencies"]:
    if re.split(r"[<>=\[]", d)[0].strip().lower() != "pyaudio":
        print(d)
PY
pip install --no-cache-dir -r deps.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir --no-deps -e <code>/fish-speech
```

Parse that array with a TOML parser. Splitting the file on `]` truncates the list at
`einx[torch]` and silently drops the nine entries after it, `descript-audio-codec` among them —
which is what provides the `audiotools` module the codec loader imports. The result installs
cleanly and then fails at model load.

## 3. What this repository needs on top

The pilot imports the F0 stack, whose dependencies are not fish-speech's:

```bash
pip install --no-cache-dir praat-parselmouth praatio matplotlib
```

## 4. Weights

```bash
python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('fishaudio/s2-pro',
      allow_patterns=['*.safetensors','*.json','codec.pth','*.jinja','LICENSE.md']))"
```

11 GB. Point `generation.fish_s2_local` at the snapshot directory that prints. The licence is
Fish Audio Research: research and non-commercial use, commercial use by separate agreement.

## 5. Verify before submitting a job

Check the imports the pipeline actually performs, not pip's exit status — a rolled-back or
truncated install reports success just the same:

```bash
python -c "
import audiotools, einx
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import init_model, load_codec_model, generate_long
print('ok')"
```

## Measured

| | |
|---|---|
| Model load + 12 generations (the pilot) | 3 min 21 s wall, 18.8 GB peak, 4 cores at 20% (job 5365458) |
| Peak memory, model load only | 18.9 GB (job 5365111, bf16 on one A100) |
| Offline Fast-AR re-run | 38 ms a frame (`pipeline/patch_backbone.py --system s2`) |

The environment also carries the F0 stack (section 3), so CPU-only analyses of S2 contours can run
in it as well as in the CSM environment.
