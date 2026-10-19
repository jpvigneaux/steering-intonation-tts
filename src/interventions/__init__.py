"""Token-space interventions on RVQ speech language models.

The probing infrastructure this builds on (the Mimi codec wrapper, the configuration layer) lives
in the companion ``basic8probes`` package. Modules here concern generation and its measurement:

``config``       the repository's configuration keys, on top of the companion package's
``archive``      per-frame token archives: every quantizer level of every sampled frame
``alignment``    the transfer-window policy that makes a cross-condition exchange comparable
``swap``         level recombination (the swap operator) and depth truncation
``depth``        CSM: re-running the depth model on stored, optionally modified, inputs
``fast_ar``      Fish S2: the same for its Fast AR
``f0``           F0 tracking, ERB conversion, and the two centering schemes
``discriminate`` the read-outs that decide which condition an output belongs to
``measure``      normalized displacement with bootstrap intervals, and word error rate
``csm``          CSM's material (sentences, conditions) and instrumented sampling
``fish_s2``      Fish S2's instrumented sampling
``prompts``      voice prompts read from the ten-speaker prompt plan

The paper's three probe-free read-outs are among ``discriminate.default_families()``: the excursion,
the velocity and the per-speaker cluster read-out.

See README.md for which script produces which result.
"""

__version__ = "0.1.0"
