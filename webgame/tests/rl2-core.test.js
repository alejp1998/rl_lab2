/**
 * Node tests for the RL Lab 2 web core (Lunar Lander + DQN).
 */
const { test } = require("node:test");
const assert = require("node:assert");

const RL2 = require("../js/rl2-core.js");

test("lander physics: gravity pulls down, main thrust counters it", () => {
  const env = RL2.createLander(1);
  const vy0 = env.vy;
  RL2.landerStep(env, 0); // no thrust
  assert.ok(env.vy > vy0, "gravity must increase vy");
  env.vy = 0.02;
  const vy1 = env.vy;
  RL2.landerStep(env, 2); // main thrust
  assert.ok(env.vy < vy1, "main thrust must reduce vy");
  assert.ok(env.fuel < 1, "thrust burns fuel");
});

test("lander: safe landing on the pad gives +100, crash gives -100", () => {
  const env = RL2.createLander(2);
  env.x = 0.5;
  env.y = 0.996; // crosses y=1 after gravity in one step
  env.vx = 0;
  env.vy = 0.005;
  const ok = RL2.landerStep(env, 0);
  assert.strictEqual(ok.reward, 100);
  assert.strictEqual(env.landed, true);

  const bad = RL2.createLander(3);
  bad.x = 0.5;
  bad.y = 0.95;
  bad.vy = 0.05; // too fast
  const crash = RL2.landerStep(bad, 0);
  assert.strictEqual(crash.reward, -100);
  assert.strictEqual(bad.crashed, true);

  const off = RL2.createLander(4);
  off.x = 0.2; // off the pad
  off.y = 0.996;
  off.vy = 0.005;
  const offPad = RL2.landerStep(off, 0);
  assert.strictEqual(offPad.reward, -100);
});

test("state vector is 5-dim and bounded-ish", () => {
  const env = RL2.createLander(5);
  const s = RL2.stateVec(env);
  assert.strictEqual(s.length, 5);
  assert.ok(Number.isFinite(s[0]) && Number.isFinite(s[4]));
});

test("NN forward shape: 5 inputs -> 4 action logits", () => {
  const nn = RL2.createNN(5, 32, 32, 4);
  const q = RL2.nnForward(nn, [0.5, 0.5, 0, 0, 1]);
  assert.strictEqual(q.length, 4);
  assert.ok(q.every(Number.isFinite));
});

test("nnCopy copies weights exactly", () => {
  const a = RL2.createNN(3, 4, 4, 2);
  const b = RL2.createNN(3, 4, 4, 2);
  RL2.nnCopy(b, a);
  assert.deepStrictEqual(b.l1.w, a.l1.w);
  assert.deepStrictEqual(b.l3.b, a.l3.b);
});

test("replay buffer caps at L and batch returns N samples", () => {
  const buf = RL2.createBuffer(100, 0, 8);
  assert.strictEqual(buf.C, Math.floor(100 / 8));
  for (let i = 0; i < 150; i++) {
    RL2.bufferAdd(buf, { s: [i], a: 0, r: 0, ns: [i + 1], done: false });
  }
  assert.strictEqual(buf.buffer.length, 100);
  assert.strictEqual(buf.buffer[0].s[0], 50); // oldest dropped
  const batch = RL2.bufferBatch(buf);
  assert.strictEqual(batch.length, 8);
});

test("DQN training step reduces TD error over iterations", () => {
  // Simple regression task: always push main thrust (action 2) toward the pad
  const dqn = RL2.createDQN({
    inputSize: 5, h1: 16, h2: 16, nActions: 4,
    L: 400, C: 0, N: 16, gamma: 0.9, alpha: 0.003, epsilon: 0.2,
  });
  // seed the buffer with experience
  for (let i = 0; i < 200; i++) {
    const env = RL2.createLander(i + 1);
    const s = RL2.stateVec(env);
    const res = RL2.landerStep(env, 2);
    const ns = RL2.stateVec(env);
    RL2.bufferAdd(dqn.buffer, { s, a: 2, r: res.reward, ns, done: res.done });
  }
  const first = RL2.dqnTrainStep(dqn);
  for (let i = 0; i < 60; i++) RL2.dqnTrainStep(dqn);
  const last = RL2.dqnTrainStep(dqn);
  assert.ok(last <= first + 0.5, `TD error should not blow up: ${first} -> ${last}`);
});

test("epsilon-greedy choose: exploitative at eps=0", () => {
  const dqn = RL2.createDQN({
    inputSize: 5, h1: 8, h2: 8, nActions: 4,
    L: 100, C: 0, N: 4, gamma: 0.9, alpha: 0.001, epsilon: 0,
  });
  // force a known preference
  dqn.online.l3.b = [0, 0, 10, 0];
  let chosen = 0;
  for (let i = 0; i < 20; i++) {
    const a = RL2.dqnChoose(dqn, [0.5, 0.5, 0, 0, 1], 0);
    if (a === 2) chosen++;
  }
  assert.strictEqual(chosen, 20, "eps=0 must always pick the max-Q action");
});
