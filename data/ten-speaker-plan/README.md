# The ten-speaker design

`prompt_plan.csv` is the prompt plan `pipeline/plan_prompts.py` drew (seed 20260921) for the
ten-speaker set: for each voice, carrier sentence and prompt, the eight human recordings given to
the model, in order (`position`), with the tune of each. `file` is a recording name in the human
corpus of Cole et al. (2023), whose own speaker codes are `speaker`. `generate.py`,
`generate_fish_s2.py`, `capture_prenorm.py` and `steer_loop.py` read it through `--prompt-plan`.

It cannot be regenerated from this repository: `plan_prompts.py` draws the speakers from the
corpus's participant survey, which is not distributed. For the same reason the plan omits the
speakers' gender. The paper reports it in aggregate (five female, four male, one non-binary),
and no script reads it.
