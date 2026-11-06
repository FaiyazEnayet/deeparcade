// Stage 3 deeparcade DA3: isolated nonvisual harness for web/game.html.
//
// Loads the real game script from web/game.html in a Node vm with stubbed
// browser globals (document, localStorage, performance, matchMedia,
// requestAnimationFrame). No DOM is rendered and no browser is needed.
// The harness replays the canonical single-step transition fixtures in
// tests/fixtures/stage3_deeparcade_3_fixtures.json (identical start state,
// stated food, one action) against the browser Snake core and compares
// body, head, heading, food, score, steps, reward, terminal flags and the
// 32-state encoding with the Python specification. With
// --print-dqn-logits it also reconstructs the frozen DQN observation
// states from tests/fixtures/dqn_parity_observations.json, checks the
// browser encodeDQN reproduces each frozen vector exactly, and prints
// the real browser dqnLogits forward pass as one DA3_DQN_LOGITS line.
//
// The committed DQN weights (runs/dqn_weights.json) are injected into
// the game script exactly as scripts/make_game.py does at build time.
// With --mutate-action-order the LEFT and RIGHT turns are swapped inside
// the loaded game script; the turning fixtures must then fail, proving
// the fixtures detect a drifted action convention.
//
// Usage: node tests/stage3_deeparcade_3_harness.js [--print-dqn-logits]
//        [--mutate-action-order]
// Exit code 0 when every check passes, 1 otherwise.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(ROOT, "web", "game.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
// The game logic is the second script block; the first is a policy stub.
let gameScript = scripts[1];
if (!gameScript || !gameScript.includes("class Snake")) {
  throw new Error("game script block with the Snake class not found in web/game.html");
}
if (!gameScript.includes("function dqnLogits(game)")) {
  throw new Error("dqnLogits not found in web/game.html; DA3 needs the logits accessor");
}
// Inject the committed DQN weights exactly as scripts/make_game.py does.
const weights = JSON.parse(fs.readFileSync(
  path.join(ROOT, "runs", "dqn_weights.json"), "utf8"));
if (!gameScript.includes("/*__DQNW__*/null")) {
  throw new Error("DQNW injection marker not found in web/game.html");
}
gameScript = gameScript.replace("/*__DQNW__*/null", JSON.stringify(weights));

const MUTATE = process.argv.includes("--mutate-action-order");
const PRINT_LOGITS = process.argv.includes("--print-dqn-logits");
if (MUTATE) {
  const src = "function relHeading(h, a){ return a===LEFT ? (h+3)%4 : a===RIGHT ? (h+1)%4 : h; }";
  const dst = "function relHeading(h, a){ return a===LEFT ? (h+1)%4 : a===RIGHT ? (h+3)%4 : h; }";
  if (gameScript.split(src).length - 1 !== 1) {
    throw new Error("relHeading definition not found exactly once; cannot mutate");
  }
  gameScript = gameScript.replace(src, dst);
}

const fixtures = JSON.parse(fs.readFileSync(
  path.join(__dirname, "fixtures", "stage3_deeparcade_3_fixtures.json"), "utf8"));

function makeCtx() {
  const store = new Map();
  const ctxStub = new Proxy(
    {},
    { get: () => () => {} }
  );
  class El {
    constructor() {
      this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
      this.style = {};
      this.textContent = "";
      this.innerHTML = "";
      this.disabled = false;
      this.value = "classic";
      this.options = [];
      this.selectedOptions = [{ disabled: false, textContent: "classic" }];
    }
    getContext() { return ctxStub; }
    addEventListener() {}
    querySelectorAll() { return []; }
    focus() {}
    click() {}
  }
  const elements = new Map();
  const sandbox = {
    console,
    Math,
    JSON,
    Map,
    Set,
    Float32Array,
    performance: { now: () => Date.now() },
    requestAnimationFrame: () => 0,
    matchMedia: () => ({ matches: false }),
    addEventListener: () => {},
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => { store.set(k, String(v)); },
      removeItem: (k) => { store.delete(k); },
    },
    document: {
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, new El());
        return elements.get(id);
      },
      querySelectorAll: () => [],
      createElement: () => new El(),
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  return vm.createContext(sandbox);
}

const ctx = makeCtx();
const exportLine =
  "\n;globalThis.__harness={Snake,RNG,dqnLogits,dqnAct," +
  "STRAIGHT,LEFT,RIGHT};\n";
vm.runInContext(gameScript + exportLine, ctx);
const H = ctx.__harness;

const failures = [];
function check(name, cond, detail) {
  if (cond) {
    console.log("ok - " + name);
  } else {
    failures.push(name);
    console.log("FAIL - " + name + (detail !== undefined ? " (" + detail + ")" : ""));
  }
}

function sameCell(a, b) {
  if (a === null || b === null) return a === b;
  return a[0] === b[0] && a[1] === b[1];
}

function reasonOf(game, result) {
  if (!game.done) return null;
  if (game.won) return "win";
  if (result.reward === -10) return "death";
  return "timeout";
}

// Replay one canonical case against the browser core with the start
// state injected and food placed exactly as stated: the initial food
// first, then food_after on an eat (null when the board fills).
function replayJS(c) {
  const g = new H.Snake(c.width, c.height, 1, "classic");
  const queue = [c.food.slice(), c.food_after ? c.food_after.slice() : null];
  g._placeFood = function () {
    const cell = queue.length ? queue.shift() : null;
    if (cell) {
      const onBody = this.body.slice(1).some((p) => p[0] === cell[0] && p[1] === cell[1]);
      if (onBody) throw new Error("stated food placed on the body in " + c.name);
      this.food = cell;
    } else {
      this.food = null;
    }
  };
  g.body = c.start.body.map((p) => p.slice());
  g.heading = c.start.heading;
  g.score = c.start.score;
  g.steps = c.start.steps;
  g.done = false;
  g.won = false;
  g._placeFood();
  const records = [snapJS(g, null, null)];
  const r = g.step(c.action);
  records.push(snapJS(g, r.reward, reasonOf(g, r)));
  return { game: g, records };
}

function snapJS(g, reward, reason) {
  return {
    body: g.body.map((p) => p.slice()),
    heading: g.heading,
    food: g.food ? g.food.slice() : null,
    score: g.score,
    steps: g.steps,
    reward,
    done: g.done,
    reason,
    state: g.encodeState(),
  };
}

function recordDiff(exp, got) {
  if (JSON.stringify(exp.body) !== JSON.stringify(got.body)) return "body";
  if (exp.heading !== got.heading) return "heading";
  if (!sameCell(exp.food, got.food)) return "food";
  if (exp.score !== got.score) return "score";
  if (exp.steps !== got.steps) return "steps";
  if (exp.reward !== got.reward) return "reward";
  if (exp.done !== got.done) return "done";
  if (exp.reason !== got.reason) return "reason";
  if (exp.state !== got.state) return "state";
  return null;
}

// 1. Fixture replays: browser transitions equal the Python specification.
for (const c of fixtures.cases) {
  let out;
  try {
    out = replayJS(c);
  } catch (e) {
    check("case " + c.name + " replays without throwing", false,
      e.constructor.name + ": " + e.message);
    continue;
  }
  check("case " + c.name + " replays without throwing", true);
  const { game, records } = out;
  let bad = null;
  if (records.length !== c.expected.length) {
    bad = "record count " + records.length + " != " + c.expected.length;
  } else {
    for (let i = 0; i < records.length; i++) {
      const d = recordDiff(c.expected[i], records[i]);
      if (d) { bad = "record " + i + " field " + d; break; }
    }
  }
  check("case " + c.name + " matches the Python transitions step for step",
    bad === null, bad || undefined);
  const lastExp = c.expected[c.expected.length - 1];
  const lastGot = records[records.length - 1];
  check("case " + c.name + " final head agrees",
    JSON.stringify(lastExp.body[lastExp.body.length - 1]) ===
    JSON.stringify(lastGot.body[lastGot.body.length - 1]));
  if (lastExp.done) {
    check("case " + c.name + " ends done", game.done === true);
    check("case " + c.name + " win flag matches the Python outcome",
      (game.won === true) === (lastExp.reason === "win"),
      "won=" + game.won + " reason=" + lastExp.reason);
    let s1, s2, threw = false;
    try { s1 = game.encodeState(); s2 = game.encodeState(); } catch (e) { threw = true; }
    check("case " + c.name + " terminal observation is consistent",
      !threw && s1 === lastExp.state && s2 === lastExp.state,
      threw ? "encodeState threw on a done game" : "state=" + s1);
    let stepThrew = false;
    try { game.step(H.STRAIGHT); } catch (e) { stepThrew = true; }
    check("case " + c.name + " refuses to step once done", stepThrew);
  }
}

// 2. DQN logits from the real browser forward pass, on the frozen
// parity observations. Each observation state is reconstructed from its
// frozen encode_dqn vector (head plane, body plane, food plane, heading
// one-hot); the browser encodeDQN must reproduce the vector exactly
// before dqnLogits runs on it.
if (PRINT_LOGITS) {
  const dqnFx = JSON.parse(fs.readFileSync(
    path.join(__dirname, "fixtures", "dqn_parity_observations.json"), "utf8"));
  const logits = [];
  let roundTripOk = true;
  for (const obs of dqnFx.observations) {
    const v = obs.vector;
    const headIdx = v.slice(0, 144).indexOf(1);
    const bodyIdx = [];
    for (let i = 0; i < 144; i++) if (v[144 + i] === 1) bodyIdx.push(i);
    const foodIdx = v.slice(288, 432).indexOf(1);
    const heading = v.slice(432, 436).indexOf(1);
    const g = new H.Snake(12, 12, 1, "classic");
    const head = [Math.floor(headIdx / 12), headIdx % 12];
    g.body = bodyIdx.map((i) => [Math.floor(i / 12), i % 12]).concat([head]);
    g.heading = heading;
    g.food = foodIdx === -1 ? null : [Math.floor(foodIdx / 12), foodIdx % 12];
    g.done = false;
    g.won = false;
    const enc = Array.from(g.encodeDQN());
    if (JSON.stringify(enc) !== JSON.stringify(v)) roundTripOk = false;
    logits.push(H.dqnLogits(g));
  }
  check("dqn observations round-trip through the browser encodeDQN",
    roundTripOk);
  check("dqnLogits returns three finite logits per observation",
    logits.length === dqnFx.observations.length &&
    logits.every((row) => row.length === 3 && row.every(Number.isFinite)));
  console.log("DA3_DQN_LOGITS " + JSON.stringify(logits));
}

if (failures.length) {
  console.log(failures.length + " check(s) failed");
  process.exit(1);
}
console.log("harness: all checks passed");
