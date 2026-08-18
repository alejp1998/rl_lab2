/**
 * RL Lab 2 — interactive cockpit: Lunar Lander with live DQN learning + manual flight.
 */
(function () {
  "use strict";

  var S = window.RL2;
  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var $id = function (id) {
    return document.getElementById(id);
  };

  var PAL = {
    dark: {
      sky1: "#0b1020",
      sky2: "#101b33",
      star: "#e2e8f0",
      ground: "#1e293b",
      groundEdge: "#334155",
      pad: "#fbbf24",
      padEdge: "#b45309",
      lander: "#e2e8f0",
      landerEdge: "#94a3b8",
      flame: "#f59e0b",
      flameHot: "#fde68a",
      text: "#e2e8f0",
      muted: "#94a3b8",
      faint: "#64748b",
      chart: "#22d3ee",
      chartGrid: "rgba(148,163,184,0.15)",
      ok: "#34d399",
      bad: "#f87171",
    },
    light: {
      sky1: "#eaf2fb",
      sky2: "#dbe8f8",
      star: "#334155",
      ground: "#d7dee9",
      groundEdge: "#94a3b8",
      pad: "#f59e0b",
      padEdge: "#b45309",
      lander: "#0f172a",
      landerEdge: "#334155",
      flame: "#f97316",
      flameHot: "#fde68a",
      text: "#0f172a",
      muted: "#475569",
      faint: "#64748b",
      chart: "#0891b2",
      chartGrid: "rgba(51,65,85,0.12)",
      ok: "#059669",
      bad: "#dc2626",
    },
  };

  function pal() {
    var t = document.documentElement.getAttribute("data-theme");
    return PAL[t === "dark" ? "dark" : "light"];
  }

  // ---------------------------------------------------------------- state
  var mode = "learn"; // learn | play
  var env = S.createLander(1);
  var dqn = null;
  var episode = 0;
  var rewards = [];
  var landings = 0;
  var crashes = 0;
  var lastAction = -1;
  var lastReward = 0;
  var epsilon = 0.9;
  var frames = 0;

  var cfg = {
    gamma: 0.99,
    alpha: 0.001,
    epsilon: 0.9,
    epsDecay: 0.996,
    L: 8000,
    N: 32,
    speed: 4,
  };

  function log(msg) {
    var box = $id("log");
    var div = document.createElement("div");
    div.textContent = "› " + msg;
    box.appendChild(div);
    while (box.children.length > 60) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  }

  function setResult(title, sub, cls) {
    var r = $id("hud-result");
    r.className = "hud-result " + cls;
    r.innerHTML =
      '<div class="hud-result-title">' +
      title +
      "</div>" +
      '<div class="hud-result-sub">' +
      sub +
      "</div>";
  }
  function clearResult() {
    $id("hud-result").className = "hud-result hidden";
  }

  function resetEpisode() {
    env = S.createLander(Math.floor(Math.random() * 1e9));
    lastAction = -1;
    clearResult();
  }

  function makeDQN() {
    dqn = S.createDQN({
      inputSize: 5,
      h1: 32,
      h2: 32,
      nActions: 4,
      L: cfg.L,
      C: 64,
      N: cfg.N,
      gamma: cfg.gamma,
      alpha: cfg.alpha,
      epsilon: cfg.epsilon,
    });
    episode = 0;
    rewards = [];
    epsilon = cfg.epsilon;
    frames = 0;
  }

  function updateHud() {
    $id("hud-mode").textContent = mode === "learn" ? "Learn" : "Fly";
    $id("hud-episode").textContent = String(episode);
    $id("hud-reward").textContent = rewards.length
      ? String(Math.round(avgReward() * 10) / 10)
      : "—";
    $id("hud-stats").textContent = landings + " / " + crashes;
    $id("hud-fuel").textContent = Math.round(env.fuel * 100) + "%";
  }

  function avgReward() {
    var tail = rewards.slice(-25);
    return (
      tail.reduce(function (a, b) {
        return a + b;
      }, 0) / tail.length
    );
  }

  // ---------------------------------------------------------------- loop
  function step() {
    var sVec = S.stateVec(env);
    var action;
    if (mode === "learn") {
      if (!dqn) makeDQN();
      action = S.dqnChoose(dqn, sVec, epsilon);
    } else {
      action = manualAction();
    }
    var res = S.landerStep(env, action);
    lastAction = action;
    lastReward = res.reward;

    if (mode === "learn") {
      S.bufferAdd(dqn.buffer, {
        s: sVec,
        a: action,
        r: res.reward,
        ns: S.stateVec(env),
        done: res.done,
      });
      // gradient steps
      for (var g = 0; g < 6; g++) S.dqnTrainStep(dqn);
    }

    if (res.done) {
      rewards.push(res.reward);
      if (res.info.landed) {
        landings++;
        if (mode === "learn" && res.reward > 50) {
          log("🛬 Landing on episode " + episode + "!");
        }
        setResult(
          "🛬 Safe landing!",
          "+100 · the pad cushioned the touchdown.",
          "win",
        );
      } else if (res.info.crashed) {
        crashes++;
        setResult("💥 Crash!", "Too fast or off the pad.", "fail");
      }
      if (mode === "learn") {
        epsilon = Math.max(0.02, epsilon * cfg.epsDecay);
      }
      episode++;
      updateHud();
      resetEpisode();
    }
  }

  function manualAction() {
    var a = 0;
    if (keys["ArrowLeft"] || keys["KeyA"]) a = 1;
    else if (keys["ArrowRight"] || keys["KeyD"]) a = 3;
    if (keys["ArrowUp"] || keys["KeyW"] || keys["Space"]) a = 2;
    return a;
  }

  var keys = {};

  // ---------------------------------------------------------------- render
  function sizeCanvas() {
    var panel = $id("stage-panel");
    var w = panel.clientWidth;
    var h = panel.clientHeight;
    var dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function render() {
    var p = pal();
    var w = canvas.width / Math.max(1, window.devicePixelRatio || 1);
    var h = canvas.height / Math.max(1, window.devicePixelRatio || 1);
    ctx.fillStyle = p.sky1;
    ctx.fillRect(0, 0, w, h);

    var gh = h;

    // stars (deterministic pseudo-random)
    for (var i = 0; i < 40; i++) {
      var sx = (i * 97) % w;
      var sy = (i * 53) % gh;
      ctx.fillStyle = p.star;
      ctx.globalAlpha = 0.25 + ((i * 7) % 10) / 20;
      ctx.fillRect(sx, sy, 1.5, 1.5);
    }
    ctx.globalAlpha = 1;

    // ground
    var groundY = gh - 14;
    ctx.fillStyle = p.ground;
    ctx.fillRect(0, groundY, w, h - groundY);
    ctx.strokeStyle = p.groundEdge;
    ctx.beginPath();
    ctx.moveTo(0, groundY + 0.5);
    ctx.lineTo(w, groundY + 0.5);
    ctx.stroke();

    // pad
    var padX = w * 0.5;
    var padW = w * 0.14;
    ctx.fillStyle = p.pad;
    ctx.fillRect(padX - padW / 2, groundY - 5, padW, 5);
    ctx.strokeStyle = p.padEdge;
    ctx.strokeRect(padX - padW / 2, groundY - 5, padW, 5);

    // lander
    var lx = env.x * w;
    var ly = groundY - env.y * (gh - 80) - 30;
    var tilt = env.vx * 60;
    ctx.save();
    ctx.translate(lx, ly);
    ctx.rotate(tilt * 0.02);

    // flame when thrusting
    if (lastAction === 2 || keys["Space"] || keys["ArrowUp"] || keys["KeyW"]) {
      var fl = 10 + Math.random() * 10;
      var fg = ctx.createLinearGradient(0, 10, 0, 10 + fl);
      fg.addColorStop(0, p.flameHot);
      fg.addColorStop(1, p.flame);
      ctx.fillStyle = fg;
      ctx.beginPath();
      ctx.moveTo(-5, 10);
      ctx.lineTo(0, 10 + fl);
      ctx.lineTo(5, 10);
      ctx.closePath();
      ctx.fill();
    }
    if (lastAction === 1 || lastAction === 3) {
      var side = lastAction === 1 ? -1 : 1;
      ctx.fillStyle = p.flame;
      ctx.beginPath();
      ctx.moveTo(side * 7, 2);
      ctx.lineTo(side * (12 + Math.random() * 6), 0);
      ctx.lineTo(side * 7, -2);
      ctx.closePath();
      ctx.fill();
    }

    // body
    ctx.fillStyle = p.lander;
    ctx.strokeStyle = p.landerEdge;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -12);
    ctx.lineTo(10, 8);
    ctx.lineTo(-10, 8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // legs
    ctx.beginPath();
    ctx.moveTo(-6, 8);
    ctx.lineTo(-12, 14);
    ctx.moveTo(6, 8);
    ctx.lineTo(12, 14);
    ctx.stroke();
    // window
    ctx.fillStyle = p.sky1;
    ctx.beginPath();
    ctx.arc(0, -2, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // velocity vector
    var vs = 14;
    ctx.strokeStyle = p.muted;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(lx + env.vx * 900, ly - env.vy * 900);
    ctx.stroke();
    ctx.setLineDash([]);

    // chart
    if (dqChart && Date.now() - lastChartT > 250) {
      dqChart.update();
      lastChartT = Date.now();
    }

    // mode + episode label
    ctx.fillStyle = p.muted;
    ctx.font = "600 12px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(
      mode === "learn"
        ? "DQN learning · ε=" +
            epsilon.toFixed(2) +
            " · copies=" +
            (dqn ? dqn.copies : 0)
        : "manual flight — ← → thrusters · ↑ main",
      12,
      24,
    );
  }

  document.addEventListener("keydown", function (e) {
    if (guideOpen) return;
    if (e.code === "KeyR") {
      resetEpisode();
      log("↻ New episode.");
      return;
    }
    keys[e.code] = true;
    if (mode === "play") {
      var names = {
        ArrowLeft: "←",
        ArrowRight: "→",
        ArrowUp: "↑",
        KeyW: "↑",
        KeyA: "←",
        KeyD: "→",
        Space: "🚀",
      };
      if (names[e.code]) markKey(names[e.code] + " — thrust");
    }
  });
  document.addEventListener("keyup", function (e) {
    keys[e.code] = false;
  });

  var lastKeyEl = null;
  function markKey(label) {
    var el = document.getElementById("last-key");
    if (el) el.textContent = label;
  }

  // ---------------------------------------------------------------- guide
  var guideOpen = false;
  function wireGuide() {
    var guide = $id("guide");
    function open() {
      guideOpen = true;
      guide.classList.remove("hidden");
    }
    function close() {
      guideOpen = false;
      guide.classList.add("hidden");
    }
    $id("btn-guide").addEventListener("click", open);
    guide.querySelectorAll("[data-close-guide]").forEach(function (el) {
      el.addEventListener("click", close);
    });
    document.addEventListener("keydown", function (e) {
      if (e.code === "Escape" && guideOpen) close();
    });
  }

  // ---------------------------------------------------------------- wiring
  function readCfg() {
    cfg.gamma = Number($id("dqn-gamma").value);
    cfg.alpha = Number($id("dqn-alpha").value);
    cfg.epsilon = Number($id("dqn-eps").value);
    cfg.epsDecay = Number($id("dqn-epsdecay").value);
    cfg.L = Number($id("dqn-L").value);
    cfg.N = Number($id("dqn-N").value);
    cfg.speed = Number($id("dqn-speed").value);
  }

  function wire() {
    ["gamma", "alpha", "eps", "epsdecay", "L", "N", "speed"].forEach(
      function (k) {
        $id("dqn-" + k).addEventListener("input", function () {
          $id("dqn-" + k + "-v").textContent =
            k === "speed" ? this.value + "×" : this.value;
          readCfg();
          if ((k === "L" || k === "N") && dqn) {
            makeDQN();
            log("🧠 Network rebuilt with new buffer config.");
          }
        });
      },
    );

    $id("btn-learn").addEventListener("click", function () {
      mode = "learn";
      $id("btn-learn").classList.add("active");
      $id("btn-play").classList.remove("active");
      if (!dqn) {
        makeDQN();
        log("🧠 DQN (5→32→32→4) with replay + target nets initialized.");
      }
      updateHud();
    });
    $id("btn-play").addEventListener("click", function () {
      mode = "play";
      $id("btn-play").classList.add("active");
      $id("btn-learn").classList.remove("active");
      resetEpisode();
      log("🎮 Manual flight — land softly on the pad.");
      updateHud();
    });

    $id("btn-theme").addEventListener("click", function () {
      var t =
        document.documentElement.getAttribute("data-theme") === "dark"
          ? "light"
          : "dark";
      document.documentElement.setAttribute("data-theme", t);
      try {
        localStorage.setItem("theme", t);
      } catch (e) {}
      applyTheme();
    });
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", function (ev) {
        if (localStorage.getItem("theme")) return;
        document.documentElement.setAttribute(
          "data-theme",
          ev.matches ? "dark" : "light",
        );
        applyTheme();
      });

    $id("btn-restart").addEventListener("click", function () {
      makeDQN();
      resetEpisode();
      log("↻ DQN reset from scratch.");
      updateHud();
    });

    window.addEventListener("resize", function () {
      sizeCanvas();
    });
  }

  function applyTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    $id("btn-theme").textContent = t === "dark" ? "☀️" : "🌙";
  }

  // ---------------------------------------------------------------- main loop
  // D3 reward chart (own component, SVG — not canvas/Pixi)
  function dqAvgData() {
    var n = rewards.length;
    var data = [];
    for (var e = 0; e < n; e++) {
      var from = Math.max(0, e - 24);
      var slice = rewards.slice(from, e + 1);
      data.push(
        slice.reduce(function (a, b) {
          return a + b;
        }, 0) / slice.length,
      );
    }
    return data;
  }
  var dqChart = null;
  var lastChartT = 0;
  function initChart() {
    var el = document.getElementById("dqn-chart");
    if (!el || !window.MiniChart) return;
    dqChart = window.MiniChart(el, {
      height: 148,
      title: "episode reward (25-ep running avg)",
      emptyText: "training…",
      pad: 10,
      getData: dqAvgData,
      color: function () {
        return pal().chart;
      },
    });
  }

  function init() {
    initChart();
    applyTheme();
    wireGuide();
    wire();
    readCfg();
    sizeCanvas();
    makeDQN();
    log("🚀 Lunar Lander loaded — DQN learning with replay + target networks.");
    log(
      "📚 Lab 2: LunarLander-v2 · NN 64-64 · L=16384 · C=L/N · N=64 (ported).",
    );
    updateHud();

    function loop() {
      var steps = mode === "learn" ? cfg.speed : 1;
      for (var i = 0; i < steps; i++) step();
      frames++;
      if (mode === "learn" && frames % 20 === 0) updateHud();
      if (mode === "play") updateHud();
      render();
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  init();
})();
