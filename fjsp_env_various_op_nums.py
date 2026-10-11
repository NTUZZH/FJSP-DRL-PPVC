import numpy as np
import numpy.ma as ma
import copy
import sys
from fjsp_env_same_op_nums import EnvState


class FJSPEnvForVariousOpNums:
    """
        a batch of fjsp environments that have various number of operations

        Instances with fewer operations are padded with dummy operations
        (all-zero processing-time rows) appended after the last job, up to the
        largest operation count of the batch (max_number_of_ops).

        PPVC extensions, mirroring FJSPEnvForSameOpNums:
          - post-operation time-lags (Adaptation 2a): lag dynamics and the
            lag-augmented completion-time bound op_ct_lb;
          - anticipatory lag channels (Adaptation 2b, use_lag_features);
          - op/station type indices carried in the state (Adaptation 1).
        With all of them off (the defaults) the environment reproduces the
        original DANIEL behaviour.
    """

    def __init__(self, n_j, n_m, use_lag_features=False, isolate_padding=False):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        :param use_lag_features: (PPVC, Adaptation 2b) if True, append the
                static-lag and remaining-lag channels to the operation
                features (op_fea_dim 10 -> 12)
        :param isolate_padding: if True, dummy operations are excluded from
                the feature normalization, their feature rows are exactly
                zero, and their predecessor/successor edges are masked, so an
                instance gets the same features in a padded batch as alone.
                False keeps the original DANIEL normalization, in which the
                dummy rows enter the per-channel variance as zeros.
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        self.use_lag_features = use_lag_features
        self.isolate_padding = isolate_padding
        self.op_fea_dim = 12 if use_lag_features else 10
        self.mch_fea_dim = 8

    def set_static_properties(self):
        """
            define static properties
        """
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)

        # [E, N]
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops],
                                       fill_value=False, dtype=bool)

        cols = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True

        a = self.mask_dummy_node[:, :, np.newaxis]
        self.dummy_mask_fea_j = np.tile(a, (1, 1, self.op_fea_dim))

        self.flag_exist_dummy_node = ~(self.env_number_of_ops == self.max_number_of_ops).all()

    def set_initial_data(self, job_length_list, op_pt_list, time_lag_list=None,
                         op_type_list=None, mch_type_list=None):
        """
            initialize the data of the instances

        :param job_length_list: the list of 'job_length' (shape [J] each)
        :param op_pt_list: the list of 'op_pt' (shape [N_k, M] each; N_k may
                differ across instances)
        :param time_lag_list: (PPVC, Adaptation 2a) optional list of per-op
                time-lag vectors (shape [N_k] each): a mandatory wait after an
                operation completes before its job successor becomes
                eligible. The machine is free during the lag. None means all
                lags are zero, which is the original DANIEL behaviour.
        :param op_type_list: (PPVC, Adaptation 1) optional list of per-op
                integer type indices (shape [N_k] each)
        :param mch_type_list: (PPVC, Adaptation 1) optional list of
                per-machine integer station-type indices (shape [M] each)
        """
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        # 异工序数环境并行化
        self.env_number_of_ops = np.array([op_pt_list[k].shape[0] for k in range(self.number_of_envs)])
        self.max_number_of_ops = np.max(self.env_number_of_ops)
        # alias used by code written for FJSPEnvForSameOpNums
        self.number_of_ops = self.max_number_of_ops

        self.set_static_properties()

        self.virtual_job_length = np.copy(self.job_length)
        self.virtual_job_length[:, -1] += self.max_number_of_ops - self.env_number_of_ops

        # [E, N, M]
        self.op_pt = np.array([np.pad(op_pt_list[k],
                                      ((0, self.max_number_of_ops - self.env_number_of_ops[k]),
                                       (0, 0)),
                                      'constant', constant_values=0)
                               for k in range(self.number_of_envs)]).astype(np.float64)

        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)

        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # --- PPVC time-lag (Adaptation 2a) --------------------------------
        # true_op_lag keeps the raw lag durations ([E, N_max], same unit as
        # true_op_pt); dummy operations get lag 0. op_lag is scaled by the
        # same slope as the pt normalization above (no shift, so zero stays
        # zero) and shares the normalized time axis with op_pt.
        self.true_op_lag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        if time_lag_list is not None:
            for k in range(self.number_of_envs):
                lag_k = np.asarray(time_lag_list[k], dtype=float)
                assert lag_k.shape[0] == self.env_number_of_ops[k], \
                    f'env {k}: {lag_k.shape[0]} lags for {self.env_number_of_ops[k]} ops'
                self.true_op_lag[k, :self.env_number_of_ops[k]] = lag_k
        self.op_lag = self.true_op_lag / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # (PPVC, Adaptation 1) static categorical type indices ([E, N_max] /
        # [E, M]) kept as raw integers; the model embeds them. Dummy
        # operations get type 0; their feature rows are zero, and the model
        # zeroes the embedding of zero rows.
        if op_type_list is None:
            self.op_type = None
        else:
            self.op_type = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=int)
            for k in range(self.number_of_envs):
                self.op_type[k, :self.env_number_of_ops[k]] = np.asarray(op_type_list[k], dtype=int)
        self.mch_type = None if mch_type_list is None else np.array(mch_type_list).astype(int)

        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = np.sum(self.process_relation, 2)
        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()
        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        # A dummy operation has no compatible machine, so the masked min/max
        # above return the mask fill value (1e20). Set them to 0 so that the
        # dummy rows cannot dominate max(op_ct_lb), which is the makespan
        # bound behind the reward. Dummy feature rows are masked before
        # normalization, so this changes no policy input.
        self.op_min_pt[self.mask_dummy_node] = 0
        self.op_max_pt[self.mask_dummy_node] = 0
        self.pt_span = self.op_max_pt - self.op_min_pt

        self.mch_min_pt = np.max(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1)

        # the estimated lower bound of the completion time of operations
        # (PPVC, Adaptation 2a) the bound satisfies
        #   ct_lb[i] >= ct_lb[i-1] + lag[i-1] + min_pt[i]
        # so each op's predecessor lag is added before the per-job cumsum.
        # This keeps the bound-based reward admissible under time-lags; with
        # zero lags the arithmetic equals the original DANIEL bound. The
        # per-job range stops at the real last op, so dummy ops stay at 0.
        self.op_ct_lb = copy.deepcopy(self.op_min_pt)

        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                first = self.job_first_op_id[k][i]
                last = self.job_last_op_id[k][i]
                seg = self.op_ct_lb[k][first:last + 1]
                seg[1:] += self.op_lag[k][first:last]
                self.op_ct_lb[k][first:last + 1] = np.cumsum(seg)

        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.virtual_job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])

        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.virtual_job_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_op_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        # old
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct machine features [E, M, 6]

        # construct Compete Tensor : [E, M, M, J]
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        # construct mch graph adjacency matrix : [E, M, M]
        self.init_mch_mask()
        self.construct_mch_features()

        self.construct_pair_features()

        # (PPVC, Adaptation 1) type indices travel with the state (None
        # outside PPVC mode, which leaves the original state unchanged)
        self.old_state.update(self.fea_j, self.op_mask,
                              self.fea_m, self.mch_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs, op_type=self.op_type, mch_type=self.mch_type)

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        # self.old_pairMessage = np.copy(self.pairMessage)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        self.initial_vars()
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.mch_queue = np.full(shape=[self.number_of_envs, self.number_of_machines,
                                        self.max_number_of_ops + 1], fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.mch_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        self.unscheduled_op_nums = np.copy(self.env_number_of_ops)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))
        # (PPVC, Adaptation 2b) remaining lag of scheduled ops at the current
        # decision instant (0 before anything is scheduled)
        self.op_remain_lag = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.max_number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)

    def step(self, actions):
        self.incomplete_env_idx = np.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(self.number_of_envs - np.sum(self.done_flag))
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job]

        if (self.reverse_process_relation[self.incomplete_env_idx, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.incomplete_env_idx, chosen_job])
        self.candidate[self.incomplete_env_idx, chosen_job] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_job] = (1 - candidate_add_flag)

        self.mch_queue[
            self.incomplete_env_idx, chosen_mch, self.mch_queue_len[self.incomplete_env_idx, chosen_mch]] = chosen_op

        self.mch_queue_len[self.incomplete_env_idx, chosen_mch] += 1

        # [E]
        chosen_op_st = np.maximum(self.candidate_free_time[self.incomplete_env_idx, chosen_job],
                                  self.mch_free_time[self.incomplete_env_idx, chosen_mch])

        self.op_ct[self.incomplete_env_idx, chosen_op] = chosen_op_st + self.op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        # (PPVC, Adaptation 2a) the job becomes ready only after the post-op
        # time-lag has elapsed; the machine is free at completion time (the
        # lag occupies the module, not the station). With zero lags both
        # lines reduce to the original DANIEL update.
        self.candidate_free_time[self.incomplete_env_idx, chosen_job] = self.op_ct[self.incomplete_env_idx, chosen_op] \
            + self.op_lag[self.incomplete_env_idx, chosen_op]
        self.mch_free_time[self.incomplete_env_idx, chosen_mch] = self.op_ct[self.incomplete_env_idx, chosen_op]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.incomplete_env_idx, chosen_job],
                                       self.true_mch_free_time[self.incomplete_env_idx, chosen_mch])
        self.true_op_ct[self.incomplete_env_idx, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op] + self.true_op_lag[self.incomplete_env_idx, chosen_op]
        self.true_mch_free_time[self.incomplete_env_idx, chosen_mch] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]

        self.current_makespan[self.incomplete_env_idx] = np.maximum(self.current_makespan[self.incomplete_env_idx],
                                                                    self.true_op_ct[
                                                                        self.incomplete_env_idx, chosen_op])

        for k, j in enumerate(self.incomplete_env_idx):
            if candidate_add_flag[k]:
                self.candidate_pt[j, chosen_job[k]] = self.unmasked_op_pt[j, chosen_op[k] + 1]
                self.candidate_process_relation[j, chosen_job[k]] = self.reverse_process_relation[j, chosen_op[k] + 1]
            else:
                self.candidate_process_relation[j, chosen_job[k]] = 1

        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        pair_free_time = self.pair_free_time[self.incomplete_env_idx]
        schedule_matrix = ma.array(pair_free_time, mask=self.candidate_process_relation[self.incomplete_env_idx])

        self.next_schedule_time[self.incomplete_env_idx] = np.min(
            schedule_matrix.reshape(self.number_of_incomplete_envs, -1), axis=1).data

        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1

        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)

        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        self.update_op_mask()

        self.mch_queue_last_op_id[self.incomplete_env_idx, chosen_mch] = chosen_op

        self.unscheduled_op_nums[self.incomplete_env_idx] -= 1

        diff = self.op_ct[self.incomplete_env_idx, chosen_op] - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        for k, j in enumerate(self.incomplete_env_idx):
            self.op_ct_lb[j][chosen_op[k]:self.job_last_op_id[j, chosen_job[k]] + 1] += diff[k]
            self.op_match_job_left_op_nums[j][
            self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= 1
            self.op_match_job_remain_work[j][
            self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= \
                self.op_mean_pt[j, chosen_op[k]]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)

        # (PPVC, Adaptation 2b) remaining lag of each scheduled op at the
        # decision instant: the lag window is [op_ct, op_ct + lag], so the
        # remaining part is clip(op_ct + lag - t, 0, lag). Unscheduled and
        # dummy ops show 0; the static op_lag channel carries their lag.
        if self.use_lag_features:
            self.op_remain_lag = self.op_scheduled_flag * np.clip(
                self.op_ct + self.op_lag - self.next_schedule_time[:, np.newaxis],
                0, self.op_lag)

        self.construct_op_features()

        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = np.array([pair_free_time[k] > self.next_schedule_time[j]
                                           for k, j in enumerate(self.incomplete_env_idx)])
        self.dynamic_pair_mask[self.incomplete_env_idx] = np.logical_or(self.dynamic_pair_mask[self.incomplete_env_idx],
                                                                        self.unavailable_pairs)

        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)

        self.update_mch_mask()

        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums[self.incomplete_env_idx] -= self.process_relation[
            self.incomplete_env_idx, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time[self.
                                           incomplete_env_idx], axis=1) - self.mch_free_time[self.incomplete_env_idx]
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag[self.incomplete_env_idx] = mch_free_flag + 0
        self.mch_waiting_time[self.incomplete_env_idx] = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work[self.incomplete_env_idx] = np.maximum(-mch_free_duration, 0)

        self.construct_mch_features()

        self.construct_pair_features()

        reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs, op_type=self.op_type, mch_type=self.mch_type)
        self.done_flag = self.done()

        return self.state, np.array(reward), self.done_flag

    def done(self):
        return self.step_count >= self.env_number_of_ops

    def construct_op_features(self):
        """
            construct operation raw features, then normalize them over the
            live (neither deleted nor dummy) operations of each instance
        """
        feature_channels = [self.op_scheduled_flag,
                            self.op_ct_lb,
                            self.op_min_pt,
                            self.pt_span,
                            self.op_mean_pt,
                            self.op_waiting_time,
                            self.op_remain_work,
                            self.op_match_job_left_op_nums,
                            self.op_match_job_remain_work,
                            self.op_available_mch_nums]
        # (PPVC, Adaptation 2b) anticipatory lag channels: the static lag an
        # op will impose once scheduled, and the lag still pending for
        # scheduled ops. Both pass through the same normalization below.
        if self.use_lag_features:
            feature_channels.append(self.op_lag)
            feature_channels.append(self.op_remain_lag)
        self.fea_j = np.stack(feature_channels, axis=2)

        if self.flag_exist_dummy_node:
            mask_all = np.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def norm_operation_features(self, mask):
        """
            z-normalize the operation features across operations, excluding
            the masked (deleted and dummy) rows from the mean. With
            isolate_padding the dummy rows are also excluded from the
            variance and end as zero rows, like deleted rows.
        """
        self.fea_j[mask] = 0
        num_delete_nodes = np.count_nonzero(mask[:, :, 0], axis=1)

        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.max_number_of_ops - num_delete_nodes

        num_left_nodes = np.maximum(num_left_nodes, 1e-8)

        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes

        # original DANIEL: only deleted rows are replaced by the mean, so
        # dummy rows enter the variance as zeros
        fill_mask = mask if self.isolate_padding else self.delete_mask_fea_j
        temp = np.where(fill_mask,
                        mean_fea_j[:, np.newaxis, :], self.fea_j)
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.max_number_of_ops / num_left_nodes)

        self.fea_j = ((temp - mean_fea_j[:, np.newaxis, :]) / \
                      (std_fea_j[:, np.newaxis, :] + 1e-8))

    def construct_mch_features(self):

        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               self.mch_remain_work,
                               self.mch_free_time,
                               self.mch_working_flag), axis=2)

        self.norm_machine_features()

    def norm_machine_features(self):
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs

        num_left_mchs = np.maximum(num_left_mchs, 1e-8)

        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs

        temp = np.where(self.delete_mask_fea_m,
                        mean_fea_m[:, np.newaxis, :], self.fea_m)
        var_fea_m = np.var(temp, axis=1)

        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = ((temp - mean_fea_m[:, np.newaxis, :]) / \
                      (std_fea_m[:, np.newaxis, :] + 1e-8))

    def construct_pair_features(self):

        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)

        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :,
                         np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]

        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work
                                                [self.env_job_idx, self.candidate],
                                                axis=-1) + 1e-8

        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / mch_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / mch_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_job_remain_work,
                                   pair_wait_time), axis=-1)

    def update_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_envs, self.max_number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1
        # with isolate_padding, a dummy op has neither predecessor nor
        # successor edge, so its zero feature row stays zero through the
        # attention layers (it cannot receive the real last op's message)
        if self.isolate_padding:
            self.op_mask[self.mask_dummy_node, 0] = 1
            self.op_mask[self.mask_dummy_node, 2] = 1

    def update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        # op 0 takes its predecessor from the last row through the circular
        # roll in the attention layer; mask that edge from the last row's
        # deletion (a no-op in practice: op 0 is a job-first op, whose
        # predecessor edge is masked from the start), as in the same-op env
        object_mask[:, 0, 0] = self.deleted_op_nodes[:, -1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)
