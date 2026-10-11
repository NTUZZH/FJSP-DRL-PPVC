import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list


def _lag_rule_route_suffixes(env, pt_stat):
    """
        per-operation remaining-route sums for the lag-aware job rules
        (environment 0, true hours), cached on the env per instance
    :param env: the production environment (after set_initial_data)
    :param pt_stat: 'mean' or 'min' processing time over compatible machines
    :return: (work_suffix [N], lag_suffix [N]) where, for an operation o of a
        job whose last operation is l, work_suffix[o] = sum of pt_stat over
        o..l and lag_suffix[o] = sum of post-op lags over o..l-1 (the lag of
        the last operation delays no successor)
    """
    cache = getattr(env, '_lag_rule_cache', None)
    if cache is None or cache[0] is not env.true_op_pt:
        cache = (env.true_op_pt, {})
        env._lag_rule_cache = cache
    if pt_stat in cache[1]:
        return cache[1][pt_stat]

    pt = env.true_op_pt[0].astype(float)          # [N, M], 0 = incompatible
    compatible = pt > 0
    if pt_stat == 'mean':
        op_pt = pt.sum(axis=1) / compatible.sum(axis=1)
    else:
        op_pt = np.where(compatible, pt, np.inf).min(axis=1)
    lag = env.true_op_lag[0].astype(float)
    work_suffix = np.zeros_like(op_pt)
    lag_suffix = np.zeros_like(op_pt)
    for first, last in zip(env.job_first_op_id[0], env.job_last_op_id[0]):
        seg = slice(first, last + 1)
        work_suffix[seg] = np.cumsum(op_pt[seg][::-1])[::-1]
        route_lag = lag[seg].copy()
        route_lag[-1] = 0.0
        lag_suffix[seg] = np.cumsum(route_lag[::-1])[::-1]
    cache[1][pt_stat] = (work_suffix, lag_suffix)
    return cache[1][pt_stat]


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining

    lag-aware rules (true, unnormalized hours from env.true_op_pt and
    env.true_op_lag; the post-operation lag delays only the job):
    SPTL: shortest processing time plus the operation's post-op lag
          (pair-level, like SPT)
    LLF: longest post-op lag first, ties by shortest processing time
          (pair-level, like SPT)
    MWKRL: most work remaining, where work = mean processing time of every
          unscheduled operation of the job plus the lags between them
          (job-level with MWKR's machine choice)
    LRPTL: longest remaining route, where route = min processing time of
          every unscheduled operation plus the lags between them
          (job-level with MWKR's machine choice). Each job is a chain, so
          its remaining critical path IS its remaining work; LRPTL differs
          from MWKRL only by using the min instead of the mean processing
          time along the route
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method in ('SPTL', 'LLF'):
        # pair-level, same eligible set and tie-break as SPT
        true_cand_pt = env.true_op_pt[0][env.candidate[0]].astype(float)   # [J, M]
        cand_lag = env.true_op_lag[0][env.candidate[0]]                    # [J]
        eligible = ~env.dynamic_pair_mask[0]
        if method == 'SPTL':
            score = true_cand_pt + cand_lag[:, np.newaxis]
            score[~eligible] = float("inf")
            action_list = np.where(score.reshape(-1) == np.min(score))[0]
        else:
            lag_pair = np.broadcast_to(cand_lag[:, np.newaxis], eligible.shape).copy()
            lag_pair[~eligible] = float("-inf")
            longest = eligible & (lag_pair == np.max(lag_pair))
            pt_pair = np.where(longest, true_cand_pt, float("inf"))
            action_list = np.where(pt_pair.reshape(-1) == np.min(pt_pair))[0]
        action = np.random.choice(action_list)
        return action

    elif method in ('MWKRL', 'LRPTL'):
        # job-level, same eligible jobs, tie-break and machine choice as MWKR
        work_suffix, lag_suffix = _lag_rule_route_suffixes(
            env, 'mean' if method == 'MWKRL' else 'min')
        route_list = work_suffix[available_ops] + lag_suffix[available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(route_list)[0])]

        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def backbone_model_suffix(model_suffix, framework='DANIEL'):
    """
        model-name suffix for a given backbone. DANIEL names are unchanged
        ('+<suffix>'); PLAN names carry the 'plan' tag ('+plan-<suffix>', or
        '+plan' for an empty suffix) so the checkpoints and result files of the
        two backbones never collide.
    """
    if framework == 'PLAN':
        return '+plan' if model_suffix == '' else '+plan-' + model_suffix
    return strToSuffix(model_suffix)


def has_plan_tag(model_name):
    """True when the last '+'-component of a model name is 'plan' or 'plan-...'."""
    last = model_name.split('+')[-1]
    return last == 'plan' or last.startswith('plan-')


def check_backbone_tag(model_name, framework='DANIEL'):
    """
        guard against loading a checkpoint into the wrong backbone: PLAN model
        names must carry the 'plan' tag and DANIEL names must not
    """
    if framework == 'PLAN' and not has_plan_tag(model_name):
        raise ValueError(f"framework=PLAN but model name '{model_name}' lacks the "
                         f"'+plan' / '+plan-<suffix>' tag")
    if framework != 'PLAN' and has_plan_tag(model_name):
        raise ValueError(f"model name '{model_name}' carries the PLAN tag but "
                         f"framework={framework}")


def backbone_file_tag(model_name, framework='DANIEL'):
    """
        tag for result files whose names do not already contain the model name:
        '' for DANIEL (names unchanged), '+<model_name>' for PLAN
    """
    return f'+{model_name}' if framework == 'PLAN' else ''


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
