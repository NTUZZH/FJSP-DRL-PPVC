from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed, backbone_model_suffix
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from ppvc_instance_generator import (ppvc_instance_generator, build_factory,
                                     save_instance, DEFAULT_FACTORY, SMALL_FACTORY,
                                     TIGHT_FACTORY)
from copy import deepcopy
import json
import os
import random
import time
import sys
from model.PPO import PPO_initialize
from model.PPO import Memory

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", configs.device_id)  # an explicit CPU-only setting in the shell wins
import torch

device = torch.device(configs.device)

# network-shape flags a fine-tuning source (--init_from) must share with the run
INIT_FROM_ARCH_KEYS = ('framework', 'use_lag_features', 'use_type_embedding',
                       'fea_j_input_dim', 'fea_m_input_dim', 'type_emb_dim',
                       'n_op_types', 'n_mch_types', 'num_heads_OAB', 'num_heads_MAB',
                       'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
                       'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob')

# multi-size validation: coarse reference makespans (SPT scale) per module
# count; the selection metric is the mean of per-size makespan / scale, so
# only the RATIO of the scales across sizes matters
SPT_SCALE = {5: 198.0, 8: 210.0, 10: 216.0, 12: 232.0, 14: 245.0,
             16: 252.0, 20: 265.0, 24: 320.0, 28: 360.0, 30: 388.0}


def vali_sizes_given(argv=None):
    """
        True when --ppvc_vali_sizes is given on the command line. The flag's
        default ('10,20') serves size-mixed training; a single-size run that
        omits the flag keeps its original single validation set at n_j.
    """
    from params import parser
    default = parser.get_default('ppvc_vali_sizes')
    parser.set_defaults(ppvc_vali_sizes=None)
    try:
        given, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    finally:
        parser.set_defaults(ppvc_vali_sizes=default)
    return given.ppvc_vali_sizes is not None


class Trainer:
    def __init__(self, config):

        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs
        self.size_mix = None          # set in the PPVC branch when --ppvc_size_mix is given
        self.ppvc_variable = False    # set in the PPVC branch when --ppvc_route_variant variable
        self.vali_multi = False       # multi-size validation (size mix, or --ppvc_vali_sizes)

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        # Patched for PyTorch>=2.3: torch.set_default_tensor_type was removed.
        # Needed because Blackwell (sm_120) requires torch>=2.7/cu128; the old
        # torch 1.11 API no longer exists. set_default_device+dtype is equivalent.
        torch.set_default_dtype(torch.float32)
        if device.type == 'cuda':
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'
        elif self.data_source == 'PPVC':
            # n_j = number of modules; the machine count is dictated by the
            # factory configuration, so n_m is overridden here
            self.ppvc_factory = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY,
                                 'tight': TIGHT_FACTORY}[config.ppvc_factory]
            self.ppvc_mch_type, _ = build_factory(self.ppvc_factory)
            self.n_m = len(self.ppvc_mch_type)
            config.n_m = self.n_m
            self.data_name = f'{self.n_j}x{self.n_m}+ppvc-{config.ppvc_mix}'
            # ---- size-mixed (multi-scale) training: guarded; empty => unchanged ----
            self.size_mix = ([int(x) for x in config.ppvc_size_mix.split(',') if x.strip()]
                             if config.ppvc_size_mix else None)
            if self.size_mix:
                tag = '-'.join(str(s) for s in self.size_mix)
                self.data_name = f'mix{tag}x{self.n_m}+ppvc-{config.ppvc_mix}'
            # ---- variable routes: modules have 18-22 ops, so instances of
            # one batch may differ in total op count; the padded env
            # FJSPEnvForVariousOpNums is used and names carry '-var' ----
            self.ppvc_variable = (config.ppvc_route_variant == 'variable')
            if self.ppvc_variable:
                assert not self.size_mix, \
                    '--ppvc_route_variant variable is not wired for --ppvc_size_mix'
                self.data_name = f'{self.data_name}-var'
            # the lag-free bound ablation is wired for the fixed-route env only
            assert not (config.bound_lag_free and self.ppvc_variable), \
                '--bound_lag_free is not wired for --ppvc_route_variant variable'
            # keep the network input width in sync with the env feature width
            # (must happen BEFORE PPO_initialize reads the config)
            if config.use_lag_features:
                config.fea_j_input_dim = 12

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        # DANIEL names are unchanged; PLAN names carry the '+plan-' tag
        self.model_name = f'{self.data_name}' \
                          f'{backbone_model_suffix(config.model_suffix, config.framework)}'
        # (transfer study) a scaled-lag validation set is stored in its own dir
        if self.data_source == 'PPVC' and config.lag_scale != 1.0:
            self.vali_data_path = f'{self.vali_data_path}-lagx{config.lag_scale:g}'
        # (fine-tuning) fail fast, before any data generation, on an
        # architecture mismatch with the source checkpoint
        if config.init_from:
            self.check_init_from_architecture()

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        if self.ppvc_variable:
            # training batches are drawn with one total op count per batch
            # (sample_training_instances), so trajectories stay equal-length
            self.env = FJSPEnvForVariousOpNums(self.n_j, self.n_m,
                                               use_lag_features=config.use_lag_features,
                                               isolate_padding=True)
        else:
            self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m,
                                            use_lag_features=config.use_lag_features,
                                            bound_lag_free=config.bound_lag_free)

        # single-size PPVC training validates at several module counts when
        # --ppvc_vali_sizes is given and names a size other than n_j; the data
        # and model names stay those of the single-size run
        if self.data_source == 'PPVC' and not self.size_mix and vali_sizes_given():
            sizes = [int(x) for x in config.ppvc_vali_sizes.split(',') if x.strip()]
            self.vali_multi = (sizes != [self.n_j])

        if self.data_source == 'PPVC' and self.size_mix:
            # size-mixed validation: one fixed env per validation module-count,
            # selection minimises the mean of per-size makespan / SPT-scale so no
            # single (larger) size dominates the metric (SPT_SCALE above).
            self.vali_multi = True
            self.vali_sizes = [int(x) for x in config.ppvc_vali_sizes.split(',') if x.strip()]
            self.vali_envs, self.vali_norm = [], []
            for sz in self.vali_sizes:
                vd = self.make_ppvc_dataset(config.vali_size,
                                            seed0=config.seed_train_vali_datagen,
                                            save_dir=None, n_modules=sz)
                env = FJSPEnvForSameOpNums(sz, self.n_m,
                                           use_lag_features=config.use_lag_features,
                                           bound_lag_free=config.bound_lag_free)
                env.set_initial_data(vd[0], vd[1], time_lag_list=vd[2],
                                     op_type_list=vd[3], mch_type_list=vd[4])
                self.vali_envs.append((sz, env))
                self.vali_norm.append(SPT_SCALE.get(sz, 216.0 * sz / 10.0))
            print(f"[size-mix] train sizes={self.size_mix}  "
                  f"vali sizes={self.vali_sizes}  norms={self.vali_norm}")
            config.vali_sizes_resolved = list(self.vali_sizes)
        elif self.vali_multi:
            # single-size training, multi-size validation: the same per-size
            # sets (seed0 + i, i < vali_size) and the same normalised selection
            # as size-mixed training; each set is written to disk under its
            # own size's name (the n_j set is self.vali_data_path)
            assert self.ppvc_vali_dir(self.n_j) == self.vali_data_path
            self.vali_sizes = sizes
            self.vali_envs, self.vali_norm, self.vali_data_paths = [], [], []
            for sz in self.vali_sizes:
                vdir = self.ppvc_vali_dir(sz)
                vd = self.make_ppvc_dataset(config.vali_size,
                                            seed0=config.seed_train_vali_datagen,
                                            save_dir=vdir, n_modules=sz)
                if self.ppvc_variable:
                    env = FJSPEnvForVariousOpNums(sz, self.n_m,
                                                  use_lag_features=config.use_lag_features,
                                                  isolate_padding=True)
                else:
                    env = FJSPEnvForSameOpNums(sz, self.n_m,
                                               use_lag_features=config.use_lag_features,
                                               bound_lag_free=config.bound_lag_free)
                env.set_initial_data(vd[0], vd[1], time_lag_list=vd[2],
                                     op_type_list=vd[3], mch_type_list=vd[4])
                self.vali_envs.append((sz, env))
                self.vali_norm.append(SPT_SCALE.get(sz, 216.0 * sz / 10.0))
                self.vali_data_paths.append(vdir)
            print(f"[vali sizes] train size={self.n_j}  vali sizes={self.vali_sizes}  "
                  f"norms={self.vali_norm}  dirs={self.vali_data_paths}")
            config.vali_sizes_resolved = list(self.vali_sizes)
        elif self.data_source == 'PPVC':
            # deterministic validation set, written to disk for the record
            vali_data = self.make_ppvc_dataset(config.vali_size,
                                               seed0=config.seed_train_vali_datagen,
                                               save_dir=self.vali_data_path)
            if self.ppvc_variable:
                # validation instances differ in op count: padded env,
                # validated by validate_envs_with_various_op_nums
                self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m,
                                                        use_lag_features=config.use_lag_features,
                                                        isolate_padding=True)
            else:
                self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m,
                                                     use_lag_features=config.use_lag_features,
                                                     bound_lag_free=config.bound_lag_free)
            self.vali_env.set_initial_data(vali_data[0], vali_data[1],
                                           time_lag_list=vali_data[2],
                                           op_type_list=vali_data[3],
                                           mch_type_list=vali_data[4])
        else:
            self.test_data = load_data_from_files(self.test_data_path)
            # validation data set
            vali_data = load_data_from_files(self.vali_data_path)

            if self.data_source == 'SD1':
                self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
            elif self.data_source == 'SD2':
                self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

            self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize()
        if config.init_from:
            self.load_init_weights()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # snapshot the full config next to the logs (experiment provenance);
        # default=str keeps a future non-JSON-native arg from killing the run
        with open(f'./train_log/{self.data_source}/config_{self.model_name}.json', 'w') as f:
            json.dump({k: v for k, v in vars(self.config).items()}, f, indent=1, default=str)

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        if self.config.init_from:
            # (fine-tuning) the quality-vs-updates curve starts at the
            # zero-shot point: validate once before any gradient step; the
            # source weights are the best-found model until beaten
            vali_result = self.validate_current()
            self.save_model()
            self.record = vali_result
            self.validation_log.append(vali_result)
            self.save_validation_log()
            tqdm.write(f'Update 0 (zero-shot from {self.config.init_from}) '
                       f'validation quality: {vali_result}')

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                if self.size_mix:
                    # draw this batch's module count; rebuild the (fixed-shape) env
                    # at that size. Batch stays size-uniform, so SameOpNums holds.
                    self.n_j = random.choice(self.size_mix)
                    self.env = FJSPEnvForSameOpNums(
                        self.n_j, self.n_m,
                        use_lag_features=self.config.use_lag_features,
                        bound_lag_free=self.config.bound_lag_free)
                dataset_job_length, dataset_op_pt, dataset_lag, dataset_op_type, dataset_mch_type \
                    = self.sample_training_instances()
                if self.data_source == 'PPVC':
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt,
                                                      time_lag_list=dataset_lag,
                                                      op_type_list=dataset_op_type,
                                                      mch_type_list=dataset_mch_type)
                else:
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor,  # [sz_b, J, M]
                                                             op_type=state.op_type_tensor,  # [sz_b, N] or None
                                                             mch_type=state.mch_type_tensor)  # [sz_b, M] or None

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.vali_multi:
                    vali_result = self.validate_size_mixed()
                elif self.data_source == "SD1" or self.ppvc_variable:
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            ep_et = time.time()
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))
        # quality-vs-updates-vs-wall-clock curve: [updates done, validation
        # quality, seconds since training start]; with --init_from the first
        # entry is the zero-shot point at update 0
        offset = 1 if self.config.init_from else 0
        if not hasattr(self, 'vali_curve'):
            self.vali_curve = []
        entry = [(len(self.validation_log) - offset) * self.validate_timestep,
                 float(self.validation_log[-1]),
                 round(time.time() - self.train_st, 3)]
        if self.vali_multi:
            # multi-size validation: the sizes that contributed, their mean
            # makespans and SPT scales (quality = mean of makespan / scale)
            entry.append({'vali_sizes': list(self.vali_sizes),
                          'mean_makespan': list(self.last_vali_makespans),
                          'norm': list(self.vali_norm)})
        self.vali_curve.append(entry)
        with open(f'./train_log/{self.data_source}/valicurve_{self.model_name}.json', 'w') as f:
            json.dump(self.vali_curve, f)

    def validate_current(self):
        """
            validation quality of the current policy, dispatched as in the
            training loop (size-mixed / various op nums / same op nums)
        """
        if self.vali_multi:
            return self.validate_size_mixed()
        if self.data_source == "SD1" or self.ppvc_variable:
            return self.validate_envs_with_various_op_nums().mean()
        return self.validate_envs_with_same_op_nums().mean()

    def check_init_from_architecture(self):
        """
            (fine-tuning) the source checkpoint must have been trained with the
            same network architecture as this run; abort with the list of
            mismatched flags otherwise. Size flags (n_j, n_m) and the factory
            may differ: the policy is size-agnostic.
        """
        src = self.config.init_from
        snap_path = f'./train_log/{self.data_source}/config_{src}.json'
        ckpt_path = f'./trained_network/{self.data_source}/{src}.pth'
        if not os.path.exists(snap_path) or not os.path.exists(ckpt_path):
            sys.exit(f'[init_from] source checkpoint or config snapshot missing: '
                     f'{ckpt_path} / {snap_path}')
        with open(snap_path) as f:
            snap = json.load(f)
        # snapshots written before --framework existed are DANIEL
        snap.setdefault('framework', 'DANIEL')
        mismatch = []
        for k in INIT_FROM_ARCH_KEYS:
            if k not in snap:
                continue
            mine = getattr(self.config, k)
            if (list(mine) if isinstance(mine, (list, tuple)) else mine) != snap[k]:
                mismatch.append(f'{k}: source={snap[k]!r} this run={mine!r}')
        if mismatch:
            sys.exit(f'[init_from] architecture of {src} does not match this run; '
                     f'aborting before training:\n  ' + '\n  '.join(mismatch))

    def load_init_weights(self):
        """
            (fine-tuning) strict load of the source weights into the policy and
            the rollout copy; the optimizer starts fresh. The source file's
            sha256 is recorded in the config snapshot.
        """
        import hashlib
        ckpt_path = f'./trained_network/{self.data_source}/{self.config.init_from}.pth'
        with open(ckpt_path, 'rb') as f:
            self.config.init_from_sha256 = hashlib.sha256(f.read()).hexdigest()
        state = torch.load(ckpt_path, map_location=device)
        self.ppo.policy.load_state_dict(state, strict=True)
        self.ppo.policy_old.load_state_dict(state, strict=True)
        print(f'[init_from] loaded {ckpt_path}')

    def scale_lags(self, meta):
        """
            (transfer study) multiply every nominal lag by --lag_scale and round
            to whole hours; identity at the default 1.0
        """
        if self.config.lag_scale != 1.0:
            meta['time_lag'] = np.rint(meta['time_lag'] * self.config.lag_scale).astype(int)
        return meta

    def ppvc_vali_dir(self, n_modules):
        """
            on-disk validation directory of a single-size PPVC run at
            n_modules (same naming as self.vali_data_path at n_j)
        """
        name = f'{n_modules}x{self.n_m}+ppvc-{self.config.ppvc_mix}'
        if self.ppvc_variable:
            name = f'{name}-var'
        path = f'./data/data_train_vali/{self.data_source}/{name}'
        if self.config.lag_scale != 1.0:
            path = f'{path}-lagx{self.config.lag_scale:g}'
        return path

    def make_ppvc_dataset(self, n_instances, seed0, save_dir=None, n_modules=None):
        """
            generate a deterministic batch of PPVC instances
            (job_length, op_pt, time_lag, op_type, mch_type lists);
            optionally persist them to disk as .fjs + .meta.json.
            n_modules overrides self.n_j (used to build per-size validation sets
            under size-mixed training); defaults to self.n_j.
        """
        nj = self.n_j if n_modules is None else n_modules
        jl_list, pt_list, lag_list, op_type_list, mch_type_list = [], [], [], [], []
        for i in range(n_instances):
            jl, pt, meta = ppvc_instance_generator(
                n_modules=nj, class_mix=self.config.ppvc_mix,
                station_counts=self.ppvc_factory, seed=seed0 + i,
                route_variant=self.config.ppvc_route_variant)
            meta = self.scale_lags(meta)
            jl_list.append(jl)
            pt_list.append(pt)
            # (A0 arm) lag-blind: the policy trains and is model-selected on a
            # world without lags; true lags re-enter only at evaluation via
            # right-shift repair. On-disk instances keep their TRUE lags.
            lag_list.append(np.zeros_like(meta['time_lag'])
                            if self.config.ppvc_lagblind else meta['time_lag'])
            op_type_list.append(meta['op_type'])
            mch_type_list.append(meta['mch_type'])
            if save_dir is not None:
                # per-file check: idempotent and self-healing (instances are
                # deterministic in seed0 + i, so re-writing is never needed)
                stem = os.path.join(save_dir, f'instance_{i:03d}')
                if not os.path.exists(stem + '.fjs'):
                    os.makedirs(save_dir, exist_ok=True)
                    save_instance(stem, jl, pt, meta)
        if save_dir is not None:
            # provenance guard: the on-disk record must match the generation
            # seed, or a later run with a different seed would silently reuse
            # stale files
            meta_path = os.path.join(save_dir, 'dataset_meta.json')
            ds_meta = {'seed0': seed0, 'n_instances': n_instances,
                       'n_modules': nj, 'ppvc_mix': self.config.ppvc_mix,
                       'ppvc_factory': self.config.ppvc_factory}
            # the route variant is recorded only for variable routes, so a
            # fixed-route record keeps its original keys (absent == 'fixed')
            if self.config.ppvc_route_variant != 'fixed':
                ds_meta['route_variant'] = self.config.ppvc_route_variant
            if self.config.lag_scale != 1.0:
                ds_meta['lag_scale'] = self.config.lag_scale
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old = json.load(f)
                assert old['seed0'] == seed0, \
                    f'validation dir {save_dir} was generated with seed0=' \
                    f'{old["seed0"]}, current run uses {seed0} — stale record'
                assert old.get('route_variant', 'fixed') == self.config.ppvc_route_variant, \
                    f'dir {save_dir} holds {old.get("route_variant", "fixed")}-route ' \
                    f'instances, current run uses {self.config.ppvc_route_variant}'
                assert old.get('lag_scale', 1.0) == self.config.lag_scale, \
                    f'dir {save_dir} holds lag_scale={old.get("lag_scale", 1.0)} ' \
                    f'instances, current run uses {self.config.lag_scale}'
                for k in ('n_modules', 'ppvc_mix', 'ppvc_factory'):
                    assert old.get(k, ds_meta[k]) == ds_meta[k], \
                        f'dir {save_dir} holds {k}={old.get(k)}, current run uses {ds_meta[k]}'
            else:
                with open(meta_path, 'w') as f:
                    json.dump(ds_meta, f, indent=1)
        return jl_list, pt_list, lag_list, op_type_list, mch_type_list

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
            (dataset_TimeLag/OpType/MchType are None outside PPVC mode)
        """
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        dataset_JobLength = []
        dataset_OpPT = []
        dataset_TimeLag = []
        dataset_OpType = []
        dataset_MchType = []
        for i in range(self.num_envs):
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)

            elif self.data_source == 'PPVC':
                # per-instance seed drawn from the (seeded) global numpy
                # stream, so resampling stays reproducible given seed_train
                inst_seed = int(np.random.randint(0, 2 ** 31 - 1))
                JobLength, OpPT, meta = ppvc_instance_generator(
                    n_modules=self.n_j, class_mix=self.config.ppvc_mix,
                    station_counts=self.ppvc_factory, seed=inst_seed,
                    route_variant=self.config.ppvc_route_variant)
                if self.ppvc_variable and i > 0:
                    # variable routes: redraw until the instance has the same
                    # total op count as the batch's first instance. All envs
                    # of the batch then finish on the same step, which keeps
                    # the PPO trajectories equal-length; each instance keeps
                    # its own per-module lengths. Seeds come from the same
                    # seeded stream, so the batch stays reproducible.
                    while OpPT.shape[0] != dataset_OpPT[0].shape[0]:
                        inst_seed = int(np.random.randint(0, 2 ** 31 - 1))
                        JobLength, OpPT, meta = ppvc_instance_generator(
                            n_modules=self.n_j, class_mix=self.config.ppvc_mix,
                            station_counts=self.ppvc_factory, seed=inst_seed,
                            route_variant=self.config.ppvc_route_variant)
                meta = self.scale_lags(meta)
                if self.config.ppvc_lag_cv > 0:
                    # lag randomization: per-op truncated lognormal factor on
                    # the nonzero lags. The generator is seeded by this
                    # instance's seed from the training stream, so the run is
                    # reproducible and draws the same instances as cv = 0.
                    from eval_stochastic_lags import realize_lags
                    meta['time_lag'] = realize_lags(
                        meta['time_lag'], self.config.ppvc_lag_cv,
                        np.random.default_rng([inst_seed, 1]))
                # (A0 arm) lag-blind training: see make_ppvc_dataset note
                dataset_TimeLag.append(np.zeros_like(meta['time_lag'])
                                       if self.config.ppvc_lagblind else meta['time_lag'])
                dataset_OpType.append(meta['op_type'])
                dataset_MchType.append(meta['mch_type'])
            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        if self.data_source != 'PPVC':
            dataset_TimeLag = dataset_OpType = dataset_MchType = None
        return dataset_JobLength, dataset_OpPT, dataset_TimeLag, dataset_OpType, dataset_MchType

    def validate_size_mixed(self):
        """
            size-mixed validation: greedy-decode each per-size validation env and
            return the mean of (per-size mean makespan / per-size SPT scale). A
            lower score means the single policy is closer to (or below) SPT across
            ALL validation sizes, so model selection rewards cross-size quality
            rather than overfitting to one module count.
        """
        self.ppo.policy.eval()
        norm_scores, makespans = [], []
        for (sz, env), norm in zip(self.vali_envs, self.vali_norm):
            if isinstance(env, FJSPEnvForVariousOpNums):
                # variable routes (single-size training only)
                self.greedy_rollout_various(env)
            else:
                state = env.reset()
                while True:
                    with torch.no_grad():
                        pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,
                                                op_mask=state.op_mask_tensor,
                                                candidate=state.candidate_tensor,
                                                fea_m=state.fea_m_tensor,
                                                mch_mask=state.mch_mask_tensor,
                                                comp_idx=state.comp_idx_tensor,
                                                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                                fea_pairs=state.fea_pairs_tensor,
                                                op_type=state.op_type_tensor,
                                                mch_type=state.mch_type_tensor)
                    action = greedy_select_action(pi)
                    state, _, done = env.step(action.cpu().numpy())
                    if done.all():
                        break
            makespans.append(float(env.current_makespan.mean()))
            norm_scores.append(float(env.current_makespan.mean()) / norm)
        self.last_vali_makespans = makespans
        self.ppo.policy.train()
        return float(np.mean(norm_scores))

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, J]
                                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor,  # [sz_b, J, M]
                                        op_type=state.op_type_tensor,  # [sz_b, N] or None
                                        mch_type=state.mch_type_tensor)  # [sz_b, M] or None

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        self.greedy_rollout_various(self.vali_env)
        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def greedy_rollout_various(self, env):
        """
            greedy decoding of every instance of env, whose instances may
            differ in operation count; the policy mode is left to the caller
        """
        state = env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx],  # [sz_b, J, M]
                                        # (PPVC variable routes) type indices of the
                                        # unfinished envs; None outside PPVC mode
                                        op_type=state.op_type_tensor[batch_idx]
                                        if state.op_type_tensor is not None else None,
                                        mch_type=state.mch_type_tensor[batch_idx]
                                        if state.mch_type_tensor is not None else None)

            action = greedy_select_action(pi)
            state, _, done = env.step(action.cpu().numpy())

            if done.all():
                break

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()