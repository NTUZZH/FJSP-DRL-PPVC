# SLAB: Time-Lag-Aware Deep Reinforcement Learning for Flexible Job-Shop Scheduling in PPVC Module Factories

Code, benchmark generator, test sets, trained policies, and result files for the paper

> **SLAB: Time-Lag-Aware Deep Reinforcement Learning for Flexible Job-Shop Scheduling in PPVC Module Factories**
> Ziheng Zhang, Dhivya Dharshini Kannan, and Wei Zhang.
> Preprint: [arXiv:2607.11725](https://doi.org/10.48550/arXiv.2607.11725), 2026.

If you use this paper, its benchmark data, or this code in your own work, please cite the paper
([BibTeX below](#citation)).

The paper studies flexible job-shop scheduling with post-operation time-lags (FJSP-TL) on the factory
floor of Prefabricated Prefinished Volumetric Construction (PPVC). After casting, waterproofing, and
painting, a module must wait for curing, a ponding test, or paint drying, and its station is free for
other modules during that wait. SLAB (Scheduling under time-Lags with an Admissible Bound) is a decision
process for dispatchers trained by deep reinforcement learning: its transition releases a module only
after its wait, and its reward is the decrease of a lower bound on the makespan that includes the waits.
The benchmark generator draws its orders from the Singapore BCA PPVC guidebook.

## Attribution and provenance

This repository is a derivative work. It adapts the DANIEL solver and ports two further scheduling
networks. Those components remain the property of, and under the license of, their original authors.
Please respect the upstream licenses and cite the original works.

- **DANIEL**: R. Wang, G. Wang, J. Sun, F. Deng, J. Chen, "Flexible Job Shop Scheduling via Dual
  Attention Network Based Reinforcement Learning," *IEEE Transactions on Neural Networks and Learning
  Systems*, 2024, DOI 10.1109/TNNLS.2023.3306421. Upstream repository:
  <https://github.com/wrqccc/FJSP-DRL>. The dual-attention policy network (`model/`), the PPO training
  loop, the base FJSP environments (`fjsp_env_*.py`), and the OR-Tools CP-SAT baseline derive from this
  work.
- **HGNN** (folder `backbones/hgnn/`): W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling
  via Graph Neural Network and Deep Reinforcement Learning," *IEEE Transactions on Industrial
  Informatics*, 19(2):1600-1610, 2023, DOI 10.1109/TII.2022.3189725. Upstream repository:
  <https://github.com/songwenas12/fjsp-drl>, Apache License 2.0. The license text and the list of
  changed and added files are in `backbones/hgnn/LICENSE` and `backbones/hgnn/NOTICE`.
- **PLAN** (file `model/plan.py`): the parallel-attention and liquid-encoder backbone of the PLAN fork of
  DANIEL, "PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in
  Flexible Job Shop Scheduling," arXiv:2608.03041, 2026. The PLAN code is not public, and its
  authors gave access to it for this work. The layers and state-dict names are kept, so the fork's
  checkpoints load unchanged.
- Further upstreams credited by DANIEL: L2D (<https://github.com/zcaicaros/L2D>), Google OR-Tools, and
  pyGAT.

With every extension flag disabled, this implementation reproduces the published DANIEL results exactly
(mean makespan 408.40 on the SD2 `10x5+mix` benchmark, 100 of 100 instances identical).

### What is original to this repository

- **Time-lag extensions of the environment and the reward**: lag-aware transition dynamics, the
  lag-augmented lower bound used as reward, optional lag feature channels, and operation and station
  type embeddings with liveness masking. All are switched by flags in `params.py`.
- **BCA-grounded PPVC benchmark generator**: `ppvc_instance_generator.py`.
- **Evaluation and analysis code**: the `eval_*.py` scripts, `right_shift_repair.py`,
  `run_cpsat_ppvc.py`, the independent feasibility validator `schedule_validator.py`, the paired
  statistics in `stats_tests.py`, and the two table generators `make_paper_macros.py` and
  `make_supplement_tables.py`.

## Repository layout

| Path | Contents |
|---|---|
| `model/` | dual-attention policy network and PPO (DANIEL-derived); `model/plan.py`, the PLAN backbone |
| `backbones/hgnn/` | HGNN scheduler (Song et al.) with a time-lag environment, training and evaluation scripts, and five trained runs |
| `fjsp_env_same_op_nums.py`, `fjsp_env_various_op_nums.py` | FJSP and FJSP-TL environments |
| `ppvc_instance_generator.py` | PPVC instance generator |
| `make_*_testsets.py` | writers of the 50- and 80-module, independent, variable-route, and scaled-wait test sets |
| `eval_*.py`, `run_cpsat_ppvc.py`, `ortools_solver.py` | evaluation of policies, dispatching rules, metaheuristics, and CP-SAT |
| `stats_tests.py` | paired Wilcoxon tests, Holm adjustment, rank-biserial effect size, bootstrap intervals |
| `make_paper_macros.py`, `make_supplement_tables.py` | every number and table of the paper and its supplement, computed from the result files |
| `schedule_validator.py` | independent feasibility check of every schedule |
| `data/PPVC/` | test sets (instance files and `dataset_meta.json` with the generator seed) |
| `data/data_train_vali/PPVC/` | the two validation sets of the SLAB policy (10 and 20 modules, halved waits) |
| `trained_network/PPVC/` | trained policies (`.pth`, about 150 KB each) |
| `test_results/PPVC/`, `or_solution/PPVC/` | per-instance result files and CP-SAT reference and anytime records |
| `train_log/PPVC/` | resolved training configurations and validation curves of the trained policies |
| `paper_inputs/` | result files of the cost measurement and of two figures, read by the table generators |
| `run_*.sh` | training and evaluation campaign drivers |
| `test_*.py` | unit and regression tests |

The DANIEL base benchmarks (SD1, SD2, BenchData), training-time logs, and solver logs are not tracked
(see `.gitignore`).

## Quick start

```bash
conda create -n fjsp python=3.11 && conda activate fjsp
pip install -r requirements-blackwell.txt        # torch, numpy, ortools, pandas, scipy, matplotlib

# greedy evaluation of the SLAB policy (seed 301) on the 10-module test set
python eval_ppvc.py --model_name 10x25+ppvc-mixed+a1lagx05-2k-s301 \
       --data_path data/PPVC/10x25+ppvc-mixed

# train a SLAB policy: lag-aware dynamics and bound, no input aids, waits halved during training
python train.py --data_source PPVC --n_j 10 --max_updates 2000 --lag_scale 0.5 \
       --use_lag_features False --use_type_embedding False \
       --vali_size 100 --ppvc_vali_sizes 10,20 --seed_train 301 --model_suffix a1lagx05-2k-s301
```

The complete flag set of every trained policy is stored in `train_log/PPVC/config_<model>.json`, and
`preflight_train.py` checks a new configuration against a stored one (parameter count and allowed
differences) before training.

## Reproducing the tables

```bash
python make_paper_macros.py          # writes paper_tables/macros_results.{tex,json} and a preview
python make_supplement_tables.py     # writes paper_tables/supplement/*.tex
```

Both scripts only read `test_results/`, `or_solution/`, `data/`, `trained_network/`, `train_log/`, and
`paper_inputs/`. Every macro carries a comment with its source files and its computation. Run
`make_paper_macros.py` first, since the supplement tables assert that each number they share with the
paper equals the value in `macros_results.json`. The GPU name in the supplement is read from
`nvidia-smi` and is left pending on a machine without it.

The result files themselves are produced by the `eval_*.py` scripts listed below. The time-limited
runs of CP-SAT and of the memetic algorithm store the CPU affinity, the load average, and the CPU share
of each solve. A record with a CPU share below 0.9 was measured under contention and is not used.

## Test sets

Instance `i` of a set is generated with seed `seed0 + i`. The 25-station factory is the default preset,
the 15-station factory the tight preset, and the 9-station factory the small preset.

| Set (`data/PPVC/`) | Modules | Stations | Orders | seed0 | Use |
|---|---:|---:|---:|---:|---|
| `10x25+ppvc-mixed` | 10 | 25 | 100 | 50000 | main test set |
| `20x25+ppvc-mixed` | 20 | 25 | 50 | 80000 | size ladder |
| `30x25+ppvc-mixed` | 30 | 25 | 20 | 90000 | size ladder |
| `40x25+ppvc-mixed` | 40 | 25 | 20 | not recorded | size ladder of the first release (no `dataset_meta.json`) |
| `50x25+ppvc-mixed` | 50 | 25 | 50 | 100000 | production-batch size |
| `80x25+ppvc-mixed` | 80 | 25 | 50 | 110000 | production-batch size |
| `10x15+ppvc-mixed` | 10 | 15 | 100 | 60000 | tight factory |
| `5x9+ppvc-mixed` | 5 | 9 | 50 | 70000 | small factory |
| `10x25+ppvc-rc_project`, `10x25+ppvc-steel_project` | 10 | 25 | 50 each | 90000, 95000 | single-shell projects of the first release (reference only; the RC seeds overlap those of the 30-module set) |
| `10x25+ppvc-rc_project-ind`, `10x25+ppvc-steel_project-ind` | 10 | 25 | 50 each | 740000, 750000 | single-shell projects (RC or steel modules only) |
| `50x25+ppvc-rc_project`, `50x25+ppvc-steel_project` | 50 | 25 | 50 each | 760000, 770000 | single-shell projects at a production-batch size |
| `10x25+ppvc-mixed-ind` | 10 | 25 | 100 | 720000 | independent confirmation |
| `20x25+ppvc-mixed-ind` | 20 | 25 | 50 | 730000 | independent confirmation |
| `50x25+ppvc-mixed-ind` | 50 | 25 | 50 | 700000 | independent confirmation |
| `80x25+ppvc-mixed-ind` | 80 | 25 | 50 | 710000 | independent confirmation |
| `10x25+ppvc-mixed-var` | 10 | 25 | 100 | 120000 | variable routes (18 to 22 operations per module) |
| `20x25+ppvc-mixed-var` | 20 | 25 | 50 | 130000 | variable routes |
| `50x25+ppvc-mixed-var` | 50 | 25 | 50 | 140000 | variable routes |
| `80x25+ppvc-mixed-var` | 80 | 25 | 50 | 150000 | variable routes |
| `10x25+ppvc-mixed-lagx1.5` | 10 | 25 | 100 | 50000 | every wait of the main set times 1.5, rounded |
| `50x25+ppvc-mixed-lagx1.5` | 50 | 25 | 50 | 100000 | every wait of the 50-module set times 1.5, rounded |

The independent sets use seed ranges that no other test set and no training run uses. They are written by
`make_independent_testsets.py`. The 50- and 80-module sets are written by `make_scale_testsets.py`, the
variable-route sets of 50 and 80 modules by `make_var_testsets.py`, and the scaled-wait sets by
`make_lagscaled_testsets.py`. Each script recreates its sets from the generator and never overwrites an
existing instance file. The two validation sets in `data/data_train_vali/PPVC/` hold 100 orders each
(seed0 100, waits halved) and are generated by `train.py`.

## Trained policies

All files are in `trained_network/PPVC/`. The name is `<training set>+<arm>-<updates>-s<seed>.pth`.

| Group | Files | What differs |
|---|---|---|
| **SLAB policy** | `10x25+ppvc-mixed+a1lagx05-2k-s301/302/303` | lag-aware dynamics and bound, no input aids, waits halved during training, 2000 updates, three seeds |
| Wait-scale dose | `10x25+ppvc-mixed+a1lagx025-2k-s301`, `...+a1lagx075-2k-s301` | waits scaled by 0.25 and 0.75 during training |
| Full waits | `10x25+ppvc-mixed+a1-2k-s301/302/303` | waits at full length during training |
| Lag-free bound | `10x25+ppvc-mixed+a1nolagbound-2k-s301/302/303` | as the full-wait arm, with a reward bound that ignores the waits |
| Lag-free training | `10x25+ppvc-mixed+a0-2k-s301/302/303` | trained without waits; evaluated by plan then right-shift repair (`eval_a0_repair.py`) |
| Input aids | `10x25+ppvc-mixed+a2-2k-s301`, `10x25+ppvc-mixed+full2k-s301/302/303`, `10x15+ppvc-mixed+full2k-s301/302/303`, `20x25+ppvc-mixed+full2k-s301` | lag features (`a2`), lag features and type embeddings (`full`) |
| Size mix | `mix10-15-20x25+ppvc-mixed+full2k-s301` | training orders of 10, 15, and 20 modules |
| Variable routes | `10x25+ppvc-mixed-var+full2k-s301` | trained on variable-route orders |
| Uncertain waits | `10x25+ppvc-mixed+full2k-cv02-s301` | waits drawn with a coefficient of variation of 0.2 during training |
| Fine-tuning | `10x15+ppvc-mixed+ft-tight-s301`, `10x15+ppvc-mixed+fthalf-tight-s301`, `10x25+ppvc-mixed+ft-lag15-s301`, `10x25+ppvc-mixed+fthalf-lag15-s301` | 400 updates starting from `10x25+ppvc-mixed+full2k-s301` (`ft`) or from the SLAB policy (`fthalf`), on the 15-station factory or with the training waits of the starting policy times 1.5 |
| PLAN backbone | `10x25+ppvc-mixed+plan-a1lagx05-2k-s301/302/303`, `...+plan-a0-2k-s301`, `...+plan-a1-2k-s301`, `...+plan-full2k-s301` | the same arms with `--framework PLAN` |
| Cost probe | `10x25+ppvc-mixed+costprobe-200-s391` | 200-update run used only to time training |
| First release | `*+a0-lagblind*`, `*+a1-bare*`, `*+a2-lagfeat*`, `*+full`, `*+full-s301/302`, `mix10-20-30x25+ppvc-mixed+full-sizemix` | 1000-update ablation arms of the first code release |

The HGNN runs are in `backbones/hgnn/save/<run>/` (`best.pt`, `last.pt`, `run_config.json`,
`train_log.json`): `hgnn-a0-s300`, `hgnn-a1-s300`, `hgnn-a0-vs1020-s300`, `hgnn-a1-vs1020-s300`, and
`hgnn-a1lagx05-vs1020-s300`.

## Evaluation scripts

| Script | What it measures |
|---|---|
| `eval_ppvc.py` | greedy and sampling rollouts of a policy, and the dispatching rules, on a test set; every schedule is validated |
| `eval_a0_repair.py` | a policy trained without waits plans on a wait-free environment, and the plan is right-shift repaired to the true waits |
| `eval_memetic.py` | memetic algorithm for the job shop with time-lags (Caumond, Lacomme, Tchernev, 2008), extended with machine assignment, under a wall-clock budget; `--fast_eval` uses the exact compiled evaluator of `fast_lpath.py`, `--pm` sets the local-search rate (the paper uses 1.0), `--instances LO:HI` runs a range of orders |
| `fast_lpath.py` | exact longest-path schedule evaluator and decoders, bit-identical to `right_shift_repair.py`; compiled with numba when it is installed, plain Python otherwise |
| `merge_memetic_parts.py` | merges the per-range result files of `eval_memetic.py --instances` into one result file |
| `eval_ga.py` | two-vector genetic algorithm for the FJSP with the same lag semantics |
| `eval_cpsat_anytime.py` | CP-SAT on the lag-aware model, cold or warm-started from a rule or policy schedule, with the incumbent trajectory over time |
| `eval_cpsat_budget.py`, `eval_cpsat_warmstart.py` | CP-SAT under short budgets, cold and warm-started from the SPT schedule |
| `eval_stochastic_lags.py` | closed-loop execution when the realized waits differ from the nominal ones (lognormal, given coefficient of variation), with policy, rule, and CP-SAT re-planning |
| `eval_reactive.py` | re-planning after six disruption types (longer wait, machine breakdown, urgent order, material shortage, processing-time deviation, cancellation) by the policy, rule re-planners (SPTL, SPT, LLF), a genetic algorithm, and right-shift repair; `--breakdown_hours` sets the outage length, `--fast_eval` the compiled GA decoder |
| `eval_residual_cpsat.py` | CP-SAT re-solving only the unstarted part of a disrupted schedule under a time budget, started from the right-shifted plan; `--match_reactive` asserts the same disrupted states as `eval_reactive.py`, `--stability_sweep` adds a penalty per changed operation |
| `eval_shell_projects.py` | single-shell test sets (seed audit, generation, evaluation, paired tests) |
| `analysis_bound_tightness.py` | ratio of the lag-augmented and the lag-free bound to the final makespan along the policy's schedules, with an admissibility check on every state |
| `analysis_stats_tables_7_9.py` | paired tests with Holm adjustment for the training options, the uncertain-wait study, and the disruption study |
| `analysis_cpsat_time_to_slab.py` | time CP-SAT needs to reach the policy's makespan, from the anytime logs |
| `eval_buffers.py` | number of lag-buffer slots that schedules use, and the makespan when the buffer is finite |
| `eval_buffer_occupancy.py` | lag-buffer occupancy of the policy and rule schedules |
| `eval_flowline.py` | makespan when every operation keeps one station of its type (flow-line routing) |
| `eval_byproducts.py` | flow time, work in progress, station utilization, and tardiness of the makespan schedules |
| `eval_sensitivity.py` | method comparison under perturbed processing times and waits |
| `eval_cost.py` | parameter count, checkpoint size, decision latency on one CPU thread, and peak memory |
| `stats_tests.py` | paired Wilcoxon signed-rank tests with Holm adjustment, rank-biserial effect sizes, and bootstrap intervals of paired differences |
| `make_paper_macros.py` | every number in the paper, with its source files and computation |
| `make_supplement_tables.py` | every table of the supplement, checked against the paper's numbers |
| `verify_pooling_invariance.py` | numerical check that deleted operations stay zero at every layer and never enter the pooled sets of the network with type embeddings |
| `analyze_training_load.py` | station utilization and the share of waiting caused by station contention, by order size and wait scale |

## Citation

```bibtex
@article{zhang2026timelag,
  title   = {{SLAB}: Time-Lag-Aware Deep Reinforcement Learning for Flexible
             Job-Shop Scheduling in {PPVC} Module Factories},
  author  = {Zhang, Ziheng and Kannan, Dhivya Dharshini and Zhang, Wei},
  journal = {arXiv preprint arXiv:2607.11725},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.11725},
  url     = {https://doi.org/10.48550/arXiv.2607.11725}
}
```

Please also cite the original DANIEL paper (see [Attribution and provenance](#attribution-and-provenance)),
whose solver this work builds on.

## License

This is a derivative work (see `NOTICE`). The DANIEL upstream repository carries no explicit license,
and the PLAN code is not public, so the code of both remains under the original authors' copyright. For that reason this
repository does not apply a blanket open-source license: doing so would misstate the rights to the
DANIEL-derived and PLAN-derived components. They are reproduced here for academic transparency and
reproducibility, with attribution above and in `NOTICE`. The HGNN code in `backbones/hgnn/` is under the
Apache License 2.0 of its upstream. The time-lag extensions, the benchmark generator, and the evaluation
code are the authors' original contributions. To reuse the DANIEL-derived or PLAN-derived components,
please contact their authors.
