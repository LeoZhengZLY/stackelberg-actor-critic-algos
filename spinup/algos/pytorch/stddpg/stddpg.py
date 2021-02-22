from copy import deepcopy
import numpy as np
import torch
import torch.autograd as autograd
from torch.optim import Adam
import gym
import time
from scipy.sparse.linalg import cg, LinearOperator, eigsh
import spinup.algos.pytorch.stddpg.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def conjugate_gradient(vec, params, b, x=None, nsteps=10, residual_tol=1e-18,
                       reg=0, device=torch.device('cpu')):
    if x is None:
        x = torch.zeros(b.shape, device=device)

    _Ax = autograd.grad(vec, params, grad_outputs=x, retain_graph=True)
    Ax = torch.cat([g.contiguous().view(-1) for g in _Ax])
    Ax += reg * x

    r = b.clone().detach() - Ax
    p = r.clone().detach()
    rsold = torch.dot(r.view(-1), r.view(-1))

    for itr in range(nsteps):
        _Ap = autograd.grad(vec, params, grad_outputs=p, retain_graph=True)
        Ap = torch.cat([g.contiguous().view(-1) for g in _Ap])
        Ap += reg * p

        alpha = rsold / torch.dot(p.view(-1), Ap.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Ap)
        rsnew = torch.dot(r.view(-1), r.view(-1))
        if rsnew < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, itr + 1


def stddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, qleader=True, decay=True, reg=0, reg_decay=1e5, rollout_iters=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data, targ_pi=True):
        o, a, r, o2, d = data['obs'].to(device), data['act'].to(device), data['rew'].to(device), data['obs2'].to(device), data['done'].to(device)

        q = ac.q(o,a)

        # Bellman backup for Q function
        # with torch.no_grad():
        q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2)) if targ_pi else ac_targ.q(o2, ac.pi(o2))
        backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs'].to(device)
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.cpu().item(), LossPi=loss_pi.cpu().item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def prepare_params(data):
        loss_q, loss_info = compute_loss_q(data, targ_pi=False)
        loss_pi = compute_loss_pi(data)

        # player 1 leader, player 2 follower
        if qleader:
            f1, f2 = loss_q, loss_pi
            p1, p2 = list(ac.q.parameters()), list(ac.pi.parameters())
            p1_lr, p2_lr = q_lr, pi_lr
            p1_size, p2_size = var_counts[1], var_counts[0]
        else:
            f1, f2 = loss_pi, loss_q
            p1, p2 = list(ac.pi.parameters()), list(ac.q.parameters())
            p1_lr, p2_lr = pi_lr, q_lr
            p1_size, p2_size = var_counts[0], var_counts[1]

        return f1, f2, p1, p2, p1_lr, p2_lr, p1_size, p2_size, loss_info

    def stac_update(data, reg):
        f1, f2, p1, p2, p1_lr, p2_lr, p1_size, p2_size, loss_info = prepare_params(data)

        D1f1 = autograd.grad(f1, p1, create_graph=True)
        D1f1_vec = torch.cat([g.contiguous().view(-1) for g in D1f1])
        D2f1 = autograd.grad(f1, p2, create_graph=True)
        D2f1_vec = torch.cat([g.contiguous().view(-1) for g in D2f1])
        D2f2 = autograd.grad(f2, p2, create_graph=True)
        D2f2_vec = torch.cat([g.contiguous().view(-1) for g in D2f2])

        x, _ = conjugate_gradient(D2f2_vec, p2, D2f1_vec.detach(), reg=reg, device=device)  # D22f2^-1 * D2f1
        _Avec = autograd.grad(D2f2_vec, p1, x, retain_graph=True, allow_unused=True)
        grad_imp = torch.cat([g.contiguous().view(-1) if g is not None else torch.Tensor([0]).to(device) for g in _Avec])
        grad_stac = D1f1_vec.detach() - grad_imp  # D1f1 - D12f2 * D22f2^-1 * D2f1

        # naive gradient update step from copg
        # def gd_optimizer(params, grad, lr):
        #     index = 0
        #     for p in params:
        #         p.data.add_(-lr * grad[index: index + p.numel()].reshape(p.shape))
        #         index += p.numel()
        #     if index != grad.numel():
        #         raise ValueError('gradient size mismatch')
        #
        # gd_optimizer(p1, grad_stac, p1_lr)
        # gd_optimizer(p2, D2f2_vec.detach(), p2_lr)

        f1.backward()
        f2.backward()
        q_optimizer.zero_grad()
        pi_optimizer.zero_grad()

        def backward(params, grad):
            index = 0
            for p in params:
                p.grad.add_(grad[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != grad.numel():
                raise ValueError('gradient size mismatch')

        backward(p1, grad_stac)
        backward(p2, D2f2_vec.detach())

        q_optimizer.step()
        pi_optimizer.step()

        # rollout
        for i in range(rollout_iters - 1):
            if qleader:
                pi_optimizer.zero_grad()
                loss_pi = compute_loss_pi(data)
                loss_pi.backward()
                pi_optimizer.step()
            else:
                q_optimizer.zero_grad()
                loss_q, loss_info = compute_loss_q(data, targ_pi=False)
                loss_q.backward()
                q_optimizer.step()

        # Record things
        logger.store(Loss1=f1.cpu().item(), Loss2=f2.cpu().item(), **loss_info,
                     StacGradNorm=torch.norm(grad_stac).cpu().item(),
                     ImplicitGradNorm=torch.norm(grad_imp).cpu().item(),
                     D1f1Norm=torch.norm(D1f1_vec).cpu().item(),
                     D2f2Norm=torch.norm(D2f2_vec).cpu().item(),
                     Regularization=reg)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def check_derivatives(data):
        f1, f2, p1, p2, p1_lr, p2_lr, p1_size, p2_size, loss_info = prepare_params(data)

        D1f1 = autograd.grad(f1, p1, create_graph=True)
        D1f1_vec = torch.cat([g.contiguous().view(-1) for g in D1f1])
        D2f2 = autograd.grad(f2, p2, create_graph=True)
        D2f2_vec = torch.cat([g.contiguous().view(-1) for g in D2f2])

        def D11f1_matvec(vec):
            """
            input:  numpy array
            output: numpy array
            """
            vec = torch.Tensor(vec).to(device)
            _Avec = autograd.grad(D1f1_vec, p1, vec, retain_graph=True)
            Avec = torch.cat([g.contiguous().view(-1) for g in _Avec])
            return np.array(Avec.cpu())

        def D22f2_matvec(vec):
            """
            input:  numpy array
            output: numpy array
            """
            vec = torch.Tensor(vec).to(device)
            _Avec = autograd.grad(D2f2_vec, p2, vec, retain_graph=True)
            Avec = torch.cat([g.contiguous().view(-1) for g in _Avec])
            return np.array(Avec.cpu())

        D11f1_lo = LinearOperator(shape=(p1_size, p1_size), matvec=D11f1_matvec)
        D22f2_lo = LinearOperator(shape=(p2_size, p2_size), matvec=D22f2_matvec)

        D11f1_eigs, _ = eigsh(D11f1_lo)
        D22f2_eigs, _ = eigsh(D22f2_lo)

        logger.store(TestD1f1Norm=torch.norm(D1f1_vec).item(),
                     TestD2f2Norm=torch.norm(D2f2_vec).item(),
                     TestD11f1MinEig=D11f1_eigs[0],
                     TestD11f1MaxEig=D11f1_eigs[-1],
                     TestD22f2MinEig=D22f2_eigs[0],
                     TestD22f2MaxEig=D22f2_eigs[-1])

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                reg_t = reg
                if decay:
                    reg_t *= np.exp(-t / reg_decay)
                stac_update(data=batch, reg=reg_t)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            batch = replay_buffer.sample_batch(batch_size)
            check_derivatives(data=batch)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('StacGradNorm', average_only=True)
            logger.log_tabular('ImplicitGradNorm', average_only=True)
            logger.log_tabular('D1f1Norm', average_only=True)
            logger.log_tabular('D2f2Norm', average_only=True)
            logger.log_tabular('Regularization', average_only=True)
            logger.log_tabular('TestD1f1Norm', average_only=True)
            logger.log_tabular('TestD2f2Norm', average_only=True)
            logger.log_tabular('TestD11f1MinEig', average_only=True)
            logger.log_tabular('TestD11f1MaxEig', average_only=True)
            logger.log_tabular('TestD22f2MinEig', average_only=True)
            logger.log_tabular('TestD22f2MaxEig', average_only=True)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('Loss1', average_only=True)
            logger.log_tabular('Loss2', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='stddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    stddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs, qleader=True)
