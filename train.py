from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
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
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)


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
            # keep the network input width in sync with the env feature width
            # (must happen BEFORE PPO_initialize reads the config)
            if config.use_lag_features:
                config.fea_j_input_dim = 12

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m,
                                        use_lag_features=config.use_lag_features)

        if self.data_source == 'PPVC' and self.size_mix:
            # size-mixed validation: one fixed env per validation module-count,
            # selection minimises the mean of per-size makespan / SPT-scale so no
            # single (larger) size dominates the metric. SPT scales are coarse
            # reference makespans; only their RATIO across sizes matters here.
            SPT_SCALE = {5: 198.0, 8: 210.0, 10: 216.0, 12: 232.0, 14: 245.0,
                         16: 252.0, 20: 265.0, 24: 320.0, 28: 360.0, 30: 388.0}
            self.vali_sizes = [int(x) for x in config.ppvc_vali_sizes.split(',') if x.strip()]
            self.vali_envs, self.vali_norm = [], []
            for sz in self.vali_sizes:
                vd = self.make_ppvc_dataset(config.vali_size,
                                            seed0=config.seed_train_vali_datagen,
                                            save_dir=None, n_modules=sz)
                env = FJSPEnvForSameOpNums(sz, self.n_m,
                                           use_lag_features=config.use_lag_features)
                env.set_initial_data(vd[0], vd[1], time_lag_list=vd[2],
                                     op_type_list=vd[3], mch_type_list=vd[4])
                self.vali_envs.append((sz, env))
                self.vali_norm.append(SPT_SCALE.get(sz, 216.0 * sz / 10.0))
            print(f"[size-mix] train sizes={self.size_mix}  "
                  f"vali sizes={self.vali_sizes}  norms={self.vali_norm}")
        elif self.data_source == 'PPVC':
            # deterministic validation set, written to disk for the record
            vali_data = self.make_ppvc_dataset(config.vali_size,
                                               seed0=config.seed_train_vali_datagen,
                                               save_dir=self.vali_data_path)
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m,
                                                 use_lag_features=config.use_lag_features)
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
                        use_lag_features=self.config.use_lag_features)
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
                if self.size_mix:
                    vali_result = self.validate_size_mixed()
                elif self.data_source == "SD1":
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
                station_counts=self.ppvc_factory, seed=seed0 + i)
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
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old = json.load(f)
                assert old['seed0'] == seed0, \
                    f'validation dir {save_dir} was generated with seed0=' \
                    f'{old["seed0"]}, current run uses {seed0} — stale record'
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
                    station_counts=self.ppvc_factory, seed=inst_seed)
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
        norm_scores = []
        for (sz, env), norm in zip(self.vali_envs, self.vali_norm):
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
            norm_scores.append(float(env.current_makespan.mean()) / norm)
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
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

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