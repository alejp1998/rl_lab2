/**
 * RL Lab 2 web core — Lunar Lander environment + DQN (port of dqn.py).
 * Pure JS: works in the browser and under node:test.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RL2 = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // =====================================================================
  // SIMPLIFIED LUNAR LANDER (2D physics, faithful reward structure)
  // =====================================================================
  // State: [x, y, vx, vy, fuel] — x,y in [0,1], pad centered at (0.5, 1)
  // Actions: 0 none, 1 left, 2 main, 3 right

  var ENV = {
    W: 1.0,
    H: 1.0,
    gravity: 0.0006,
    mainThrust: 0.0022,
    lateralThrust: 0.0013,
    fuelCost: 0.0008,
    maxSteps: 400,
    padHalf: 0.12,
    safeVy: 0.012,
    safeVx: 0.012,
  };

  function createLander(seed) {
    var rng = mulberry32(seed || 1);
    return {
      x: 0.15 + rng() * 0.7,
      y: 0.2 + rng() * 0.15,
      vx: (rng() - 0.5) * 0.02,
      vy: 0.02 + rng() * 0.02,
      fuel: 1.0,
      steps: 0,
      rng: rng,
      crashed: false,
      landed: false,
    };
  }

  function landerStep(env, action) {
    env.steps++;
    var reward = -1; // per-step cost
    var done = false;
    var info = {};
    var thrusting = false;
    if (action === 1 && env.fuel > 0) {
      env.vx -= ENV.lateralThrust;
      env.fuel = Math.max(0, env.fuel - ENV.fuelCost * 0.6);
      thrusting = true;
    } else if (action === 3 && env.fuel > 0) {
      env.vx += ENV.lateralThrust;
      env.fuel = Math.max(0, env.fuel - ENV.fuelCost * 0.6);
      thrusting = true;
    } else if (action === 2 && env.fuel > 0) {
      env.vy -= ENV.mainThrust;
      env.fuel = Math.max(0, env.fuel - ENV.fuelCost);
      thrusting = true;
    }

    env.vy += ENV.gravity;
    env.x += env.vx;
    env.y += env.vy;

    // Mild hover shaping (like the real Lander's leg-contact bonus): reward
    // being over the pad with low descent speed — teaches controlled descent.
    var nearPad = 1 - Math.min(1, Math.abs(env.x - 0.5) / 0.5);
    var calm = 1 - Math.min(1, Math.abs(env.vy) / 0.05);
    reward += 0.05 * nearPad * calm;

    // ground contact
    if (env.y >= 1) {
      env.y = 1;
      var onPad = Math.abs(env.x - 0.5) < ENV.padHalf;
      if (onPad && Math.abs(env.vy) < ENV.safeVy && Math.abs(env.vx) < ENV.safeVx) {
        reward = 100;
        env.landed = true;
        info.landed = true;
      } else {
        reward = -100;
        env.crashed = true;
        info.crashed = true;
      }
      done = true;
    }
    if (env.steps >= ENV.maxSteps) done = true;
    if (env.fuel <= 0) reward -= 0.2; // mild fuel-out penalty each step

    return { reward: reward, done: done, info: info, thrusting: thrusting };
  }

  /** Normalized 5-dim state vector for the network. */
  function stateVec(env) {
    return [env.x, env.y, env.vx * 25, env.vy * 25, env.fuel];
  }

  // =====================================================================
  // MINI NEURAL NET (ReLU MLP — matches the lab's NN(64,64) structure)
  // =====================================================================

  function createNN(inputSize, h1, h2, outputSize) {
    // Xavier-ish init
    function layer(nIn, nOut) {
      var w = [];
      var s = Math.sqrt(2 / nIn);
      for (var i = 0; i < nOut; i++) {
        var row = [];
        for (var j = 0; j < nIn; j++) row.push((Math.random() * 2 - 1) * s);
        w.push(row);
      }
      var b = new Array(nOut).fill(0);
      return { w: w, b: b };
    }
    return {
      l1: layer(inputSize, h1),
      l2: layer(h1, h2),
      l3: layer(h2, outputSize),
    };
  }

  function nnForward(nn, x) {
    function affine(ly, v) {
      var out = [];
      for (var i = 0; i < ly.b.length; i++) {
        var acc = ly.b[i];
        var row = ly.w[i];
        for (var j = 0; j < v.length; j++) acc += row[j] * v[j];
        out.push(acc);
      }
      return out;
    }
    function relu(v) {
      return v.map(function (z) { return z > 0 ? z : 0; });
    }
    var h1 = relu(affine(nn.l1, x));
    var h2 = relu(affine(nn.l2, h1));
    return affine(nn.l3, h2); // logits per action
  }

  /** Copy weights from `src` into `dst` (target network sync). */
  function nnCopy(dst, src) {
    ["l1", "l2", "l3"].forEach(function (ly) {
      for (var i = 0; i < src[ly].w.length; i++) {
        dst[ly].w[i] = src[ly].w[i].slice();
        dst[ly].b[i] = src[ly].b[i];
      }
    });
  }

  // =====================================================================
  // EXPERIENCE REPLAY BUFFER (port of ExpRepBuffer)
  // =====================================================================

  function createBuffer(L, C, N) {
    return {
      L: L,
      C: C === 0 ? Math.floor(L / N) : C,
      N: N,
      buffer: [],
    };
  }

  function bufferAdd(buf, z) {
    if (buf.buffer.length >= buf.L) buf.buffer.shift();
    buf.buffer.push(z);
  }

  function bufferBatch(buf) {
    var n = buf.N;
    var batch = [];
    for (var i = 0; i < n; i++) {
      batch.push(buf.buffer[Math.floor(Math.random() * buf.buffer.length)]);
    }
    return batch;
  }

  // =====================================================================
  // DQN TRAINING (Adam-lite; replay + target net like the lab)
  // =====================================================================

  function createDQN(cfg) {
    var dqn = {
      cfg: cfg,
      online: createNN(cfg.inputSize, cfg.h1, cfg.h2, cfg.nActions),
      target: createNN(cfg.inputSize, cfg.h1, cfg.h2, cfg.nActions),
      buffer: createBuffer(cfg.L, cfg.C, cfg.N),
      steps: 0,
      copies: 0,
      gamma: cfg.gamma,
      alpha: cfg.alpha,
      epsilon: cfg.epsilon,
      double: !!cfg.double,
      // Adam-lite moments
      m: null,
      v: null,
      t: 0,
    };
    nnCopy(dqn.target, dqn.online);
    dqn.m = initMoments(dqn.online);
    dqn.v = initMoments(dqn.online);
    return dqn;
  }

  function initMoments(nn) {
    var m = {};
    ["l1", "l2", "l3"].forEach(function (ly) {
      m[ly] = { w: nn[ly].w.map(function (r) { return r.map(function () { return 0; }); }), b: nn[ly].b.map(function () { return 0; }) };
    });
    return m;
  }

  function dqnChoose(dqn, s, eps) {
    if (Math.random() < eps) return Math.floor(Math.random() * dqn.cfg.nActions);
    var q = nnForward(dqn.online, s);
    var best = 0;
    for (var a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
    return best;
  }

  /** One Adam-lite gradient step on a batch. Returns the mean TD error. */
  function dqnTrainStep(dqn) {
    var buf = dqn.buffer;
    if (buf.buffer.length < buf.N * 2) return 0;
    var batch = bufferBatch(buf);
    var tdSum = 0;

    // accumulate gradients per layer
    var g = initMoments(dqn.online);
    for (var z = 0; z < batch.length; z++) {
      var exp = batch[z];
      var s = exp.s;
      var a = exp.a;
      var r = exp.r;
      var ns = exp.ns;
      var done = exp.done;

      var qAll = nnForward(dqn.online, s);
      var targetQ = nnForward(dqn.target, ns);
      var maxNext;
      if (dqn.double) {
        // Double DQN: pick the action with the ONLINE net, value it with the
        // target net — cuts overestimation, learns faster and more stably
        var qAllNext = nnForward(dqn.online, ns);
        var aStar = 0;
        for (var ai = 1; ai < qAllNext.length; ai++)
          if (qAllNext[ai] > qAllNext[aStar]) aStar = ai;
        maxNext = targetQ[aStar];
      } else {
        maxNext = Math.max.apply(null, targetQ);
      }
      var td = r + (done ? 0 : dqn.gamma * maxNext) - qAll[a];
      // clip the TD error (Huber-style stabilisation, standard DQN practice)
      td = Math.max(-20, Math.min(20, td));

      // backprop through the MLP for this single sample (target = one-hot at a)
      backpropAccumulate(dqn.online, g, s, a, td);
      tdSum += Math.abs(td);
    }

    // Adam update
    dqn.t++;
    var b1 = 0.9, b2 = 0.999, epsA = 1e-8;
    ["l1", "l2", "l3"].forEach(function (ly) {
      var gLy = g[ly];
      var mLy = dqn.m[ly];
      var vLy = dqn.v[ly];
      for (var i = 0; i < gLy.w.length; i++) {
        for (var j = 0; j < gLy.w[i].length; j++) {
          mLy.w[i][j] = b1 * mLy.w[i][j] + (1 - b1) * gLy.w[i][j];
          vLy.w[i][j] = b2 * vLy.w[i][j] + (1 - b2) * gLy.w[i][j] * gLy.w[i][j];
          var mh = mLy.w[i][j] / (1 - Math.pow(b1, dqn.t));
          var vh = vLy.w[i][j] / (1 - Math.pow(b2, dqn.t));
          dqn.online[ly].w[i][j] -= (dqn.alpha * mh) / (Math.sqrt(vh) + epsA);
        }
        mLy.b[i] = b1 * mLy.b[i] + (1 - b1) * gLy.b[i];
        vLy.b[i] = b2 * vLy.b[i] + (1 - b2) * gLy.b[i] * gLy.b[i];
        var mb = mLy.b[i] / (1 - Math.pow(b1, dqn.t));
        var vb = vLy.b[i] / (1 - Math.pow(b2, dqn.t));
        dqn.online[ly].b[i] -= (dqn.alpha * mb) / (Math.sqrt(vb) + epsA);
      }
    });

    // target network sync every C steps
    dqn.steps++;
    if (dqn.steps % dqn.buffer.C === 0) {
      nnCopy(dqn.target, dqn.online);
      dqn.copies++;
    }
    return tdSum / batch.length;
  }

  /** Accumulates the gradient of (Q(s,a) - td)^2 w.r.t. weights. */
  function backpropAccumulate(nn, g, s, a, td) {
    // forward
    function affine(ly, v) {
      var out = [];
      for (var i = 0; i < ly.b.length; i++) {
        var acc = ly.b[i];
        var row = ly.w[i];
        for (var j = 0; j < v.length; j++) acc += row[j] * v[j];
        out.push(acc);
      }
      return out;
    }
    var z1 = affine(nn.l1, s);
    var h1 = z1.map(function (v) { return v > 0 ? v : 0; });
    var z2 = affine(nn.l2, h1);
    var h2 = z2.map(function (v) { return v > 0 ? v : 0; });
    var z3 = affine(nn.l3, h2);

    // dL/dz3 = -2 * td * onehot(a)  (L = td^2, td = target - Q)
    var d3 = new Array(z3.length).fill(0);
    d3[a] = -2 * td;
    // l3
    for (var i = 0; i < z3.length; i++) {
      g.l3.b[i] += d3[i];
      for (var j = 0; j < h2.length; j++) g.l3.w[i][j] += d3[i] * h2[j];
    }
    // d2
    var d2 = new Array(z2.length).fill(0);
    for (var i2 = 0; i2 < z2.length; i2++) {
      if (z2[i2] > 0) {
        for (var k = 0; k < z3.length; k++) d2[i2] += d3[k] * nn.l3.w[k][i2];
      }
    }
    for (var i3 = 0; i3 < z2.length; i3++) {
      g.l2.b[i3] += d2[i3];
      for (var j2 = 0; j2 < h1.length; j2++) g.l2.w[i3][j2] += d2[i3] * h1[j2];
    }
    // d1
    var d1 = new Array(z1.length).fill(0);
    for (var i4 = 0; i4 < z1.length; i4++) {
      if (z1[i4] > 0) {
        for (var k2 = 0; k2 < z2.length; k2++) d1[i4] += d2[k2] * nn.l2.w[k2][i4];
      }
    }
    for (var i5 = 0; i5 < z1.length; i5++) {
      g.l1.b[i5] += d1[i5];
      for (var j3 = 0; j3 < s.length; j3++) g.l1.w[i5][j3] += d1[i5] * s[j3];
    }
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  return {
    ENV: ENV,
    createLander: createLander,
    landerStep: landerStep,
    stateVec: stateVec,
    createNN: createNN,
    nnForward: nnForward,
    nnCopy: nnCopy,
    createBuffer: createBuffer,
    bufferAdd: bufferAdd,
    bufferBatch: bufferBatch,
    createDQN: createDQN,
    dqnChoose: dqnChoose,
    dqnTrainStep: dqnTrainStep,
  };
});
