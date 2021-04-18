from spinup.utils.run_utils import ExperimentGrid
from spinup import stac_pytorch as stac
from spinup import ddpg_pytorch as ddpg
from spinup import stddpg_pytorch as stddpg
from spinup import sac_pytorch as sac
from spinup import stsac_pytorch as stsac
import torch
import time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pi-lr', type=float, default=1e-3)
    parser.add_argument('--q-lr', type=float, default=1e-3)
    parser.add_argument('--reg', type=int, default=0)
    parser.add_argument('--rollout-iters', type=int, default=1)
    parser.add_argument('--decay', dest='decay', action='store_true')
    parser.add_argument('--no-decay', dest='decay', action='store_false')
    parser.set_defaults(decay=True)
    args = parser.parse_args()

    start_time = time.time()

    # eg = ExperimentGrid(name='stac')
    #
    # eg.add('env_name', 'CartPole-v0', '', True)
    # eg.add('steps_per_epoch', 4000)
    # eg.add('epochs', 200)
    # eg.add('seed', [args.seed])
    # eg.add('vanilla', [True, False], None, True)
    # eg.add('pi_lr', [1e-1], None, True)    # 1e-1, 5e-2
    # eg.add('vf_lr', [1e-2], None, True)
    # eg.add('train_v_iters', [1, 80], None, True)
    # eg.add('noise', [5e-2], None, True)

    # eg.run(stac, num_cpu=4)

    # eg = ExperimentGrid(name='ddpg')
    # eg.add('env_name', args.env, '', True)
    # eg.add('seed', [args.seed])
    # eg.add('targ_pi', [False], None, True)
    # eg.add('pi_lr', [args.pi_lr], None, True)
    # eg.add('q_lr', [args.q_lr], None, True)
    # eg.run(ddpg)

    eg = ExperimentGrid(name='stddpg')
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [args.seed])
    eg.add('qleader', [False, True], None, True)
    eg.add('pi_lr', [args.pi_lr], None, True)
    eg.add('q_lr', [args.q_lr], None, True)
    eg.add('reg', [args.reg], None, True)
    eg.add('decay', [args.decay], None, True)
    eg.add('rollout_iters', [args.rollout_iters], None, True)
    eg.run(stddpg)

    # eg = ExperimentGrid(name='sac')
    # eg.add('env_name', args.env, '', True)
    # eg.add('seed', [args.seed])
    # eg.add('pi_lr', [args.pi_lr], None, True)
    # eg.add('q_lr', [args.q_lr], None, True)
    # eg.run(sac)

    # eg = ExperimentGrid(name='stsac')
    # eg.add('env_name', args.env, '', True)
    # eg.add('seed', [args.seed])
    # eg.add('qleader', [False, True], None, True)
    # eg.add('pi_lr', [args.pi_lr], None, True)
    # eg.add('q_lr', [args.q_lr], None, True)
    # eg.add('reg', [args.reg], None, True)
    # eg.add('decay', [args.decay], None, True)
    # eg.add('rollout_iters', [args.rollout_iters], None, True)
    # eg.run(stsac)

    elapsed_time = time.time() - start_time
    print("total used time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

