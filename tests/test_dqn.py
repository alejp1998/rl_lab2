"""Unit tests for the RL Lab 2 DQN code (rl_lab2, problem 1)."""

import dqn
import numpy as np
import pytest
import torch


def test_running_average_smoothes():
    x = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
    y = dqn.running_average(x, 3)
    assert len(y) == len(x)
    assert y[-1] == pytest.approx(10.0)
    assert y[2] == pytest.approx(10 / 3)


def test_running_average_short_input_is_zero():
    y = dqn.running_average(np.array([1.0, 2.0]), 5)
    assert (y == 0).all()


def test_exp_replay_buffer_caps_at_L():
    buf = dqn.ExpRepBuffer(L=50, C=0, N=10)
    assert buf.C == 5  # default target period = L / N
    for i in range(100):
        buf.add_exp((i, 0, 0.0, i + 1, False))
    assert len(buf.buffer) == 50
    assert buf.buffer[0][0] == 50  # oldest dropped


def test_exp_replay_buffer_batch_shapes():
    buf = dqn.ExpRepBuffer(L=100, C=0, N=8)
    for i in range(100):
        buf.add_exp((np.zeros(8), 2, -1.0, np.zeros(8), False))
    states, actions, rewards, next_states, dones = buf.random_batch()
    assert states.shape == (8, 8)
    assert actions.shape == (8, 1)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 8)
    assert dones.shape == (8,)


def test_neural_network_forward_shape():
    nn = dqn.NeuralNetwork(input_size=8, hidden_size_1=64, hidden_size_2=64, output_size=4)
    out = nn.forward(torch.zeros(3, 8))
    assert out.shape == (3, 4)


@pytest.mark.slow
def test_dqn_trains_a_few_cartpole_episodes():
    """End-to-end smoke: the lab's dqn() trains on a simple env (LunarLander
    needs box2d which is not always buildable; CartPole exercises the same code)."""
    import gym

    env = gym.make("CartPole-v1")
    nn = dqn.NeuralNetwork(4, 64, 64, 2)
    nn_target = dqn.NeuralNetwork(4, 64, 64, 2)
    B = dqn.ExpRepBuffer(L=500, C=0, N=16)
    rewards, steps, eps = dqn.dqn(
        env, nn, nn_target, B,
        gamma=0.99, alpha=0.001, epsilon=0.5, n_episodes=5, max_iters=200,
    )
    env.close()
    assert len(rewards) == 5
    assert len(steps) == 5
    assert len(eps) == 5
    assert all(np.isfinite(rewards))
