import numpy as np
import torch
import torch.autograd as autograd
from torch.optim import Adam
import gym
import time
from scipy.sparse.linalg import cg, LinearOperator
import spinup.algos.pytorch.stac.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class STACBuffer:
    """
    A buffer for storing trajectories experienced by a STAC agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def stac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=1e-1,
        vf_lr=1e-2, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10, vanilla=False, reg=0, noise=0):
    """
    Stackelberg Actor Critic
    (with GAE-Lambda for advantage estimation)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to STAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = STACBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing STAC policy loss and gradient Dpfp
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = (logp * adv).mean()    # adv or ac.v(obs) ?

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss and gradient Dvfv, Dvvfv
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']

        return ((ac.v(obs) - ret)**2).mean()
        # return ((ac.v(obs) - ret)**2)[0]

    # Set up surrogate function for computing gradient of policy loss
    # with respect to value parameter Dvfp
    def compute_loss_pi_surr(data):
        obs = data['obs']
        return (ac.v(obs)).mean()

    # Set up surrogate function for computing gradient of value loss
    # with respect to policy parameter and value parameter Dpvfv
    def compute_loss_v_surr(data):
        obs, act, ret, adv = data['obs'], data['act'], data['ret'], data['adv']

        pi, logp = ac.pi(obs, act)
        dQsquared = ((ac.v(obs) - ret)**2)
        
        # return (logp*dQsquared).mean()
        # return (logp * dQsquared.mean()).mean()
        return 2 * (logp * adv).mean() * (ret[0] - ac.v(obs)[0])

    # Set up optimizers for policy and value function
    # pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    # vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        # why is this different from later ?
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of Stackelberg descent

        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.detach()
                    p.grad.zero_()

        # pi_optimizer.zero_grad()
        zero_grad(ac.pi.parameters())

        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi_surr = compute_loss_pi_surr(data)
        Dpfp = autograd.grad(loss_pi, ac.pi.parameters(), create_graph=True)
        Dpfp_vec = torch.cat([g.contiguous().view(-1) for g in Dpfp])
        Dvfp = autograd.grad(loss_pi_surr, ac.v.parameters(), create_graph=True)
        Dvfp_vec = torch.cat([g.contiguous().view(-1) for g in Dvfp])

        loss_v = compute_loss_v(data)
        loss_v_surr = compute_loss_v_surr(data)
        Dvfv = autograd.grad(loss_v, ac.v.parameters(), create_graph=True)
        Dvfv_vec = torch.cat([g.contiguous().view(-1) for g in Dvfv])
        Dvfv_surr = autograd.grad(loss_v_surr, ac.v.parameters(), create_graph=True)    # for Dpvfv
        Dvfv_surr_vec = torch.cat([g.contiguous().view(-1) for g in Dvfv_surr])

        def Dvvfv_matvec(vec):
            """
            input:  numpy array
            output: numpy array
            """
            vec = torch.Tensor(vec)
            _Avec = autograd.grad(Dvfv_vec, ac.v.parameters(), vec, retain_graph=True)
            Avec = torch.cat([g.contiguous().view(-1) for g in _Avec])
            Avec += reg * vec
            return np.array(Avec)

        def Dpvfv(vec):
            """
            input:  numpy array
            output: torch tensor
            """
            vec = torch.Tensor(vec)
            _Avec = autograd.grad(Dvfv_surr_vec, ac.pi.parameters(), vec, retain_graph=True)
            return torch.cat([g.contiguous().view(-1) for g in _Avec])

        Dvvfv_lo = LinearOperator(shape=(var_counts[1],var_counts[1]), matvec=Dvvfv_matvec)
        w, _ = cg(Dvvfv_lo, Dvfp_vec.detach().numpy(), maxiter=10)    # Dvvfv^-1 * Dvfp
        grad_imp = Dpvfv(w)
        # grad_imp = Dpvfv(w) - 2 * Dpfp_vec.detach() * (torch.dot(torch.Tensor(w), Dvfp_vec.detach()))
        grad_p = Dpfp_vec.detach() - grad_imp    # Dpfp - Dpvfv * Dvvfv^-1 * Dvfp
        if vanilla:
            grad_p = Dpfp_vec.detach()

        # loss_pi.backward()
        # mpi_avg_grads(ac.pi)    # average grads across MPI processes
        # pi_optimizer.step()

        # average grads across MPI processes
        def mpi_avg_tensor(vec):
            vec_numpy = vec.numpy()    # numpy view of tensor data
            avg_vec = mpi_avg(vec)
            vec_numpy[:] = avg_vec[:]

        # is this mathematically the best way to average over multiple processors ?
        mpi_avg_tensor(grad_p)

        # naive gradient update step from copg
        def gd_optimizer(params, grad, lr):
            index = 0
            for p in params:
                p.data.add_(-lr * grad[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != grad.numel():
                raise ValueError('gradient size mismatch')

        gd_optimizer(ac.pi.parameters(), -grad_p + noise * torch.randn(grad_p.shape), pi_lr)

        # Value function learning
        for i in range(train_v_iters):
            # vf_optimizer.zero_grad()
            # loss_v = compute_loss_v(data)
            # loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            # vf_optimizer.step()

            zero_grad(ac.v.parameters())
            loss_v = compute_loss_v(data)
            Dvfv = autograd.grad(loss_v, ac.v.parameters())
            Dvfv_vec = torch.cat([g.contiguous().view(-1) for g in Dvfv])
            mpi_avg_tensor(Dvfv_vec)
            gd_optimizer(ac.v.parameters(), Dvfv_vec + noise * torch.randn(Dvfv_vec.shape), vf_lr)

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=-pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=-(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     StacGradNorm=torch.norm(grad_p).item(),
                     VanillaGradNorm=torch.norm(Dpfp_vec).item(),
                     ImplicitGradNorm=torch.norm(grad_imp).item())

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform STAC update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('StacGradNorm', average_only=True)
        logger.log_tabular('VanillaGradNorm', average_only=True)
        logger.log_tabular('ImplicitGradNorm', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='stac_test')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    stac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, vanilla=False)