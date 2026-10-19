#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
# hlh, second design: the vector is hlh against lll alone
# (human/csm/direction-hlh-vs-lll), applied only to CSM assertions whose unsteered in-loop run the
# nearest-centroid labelling puts in lll (matrix/results/hlh2_targets.csv: 43 over ten voices).
# Sizing as queue_loso_hlh.sh: csm 5.2 s/run, load 114 s, 12.25 GB. At most 6 targets x 6 alphas =
# 36 runs: 114 + 36 x 5.2 = 301 s x1.4 = 421 s, slow-node allowance -> 10 min; 14G. Align: 6 sets
# of at most 6 -> ~8 min x1.4 -> 11 min; 2700M.
