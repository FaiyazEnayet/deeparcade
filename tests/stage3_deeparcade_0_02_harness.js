// Stage 3 deeparcade P0-02: isolated nonvisual harness for web/game.html.
//
// Loads the real game script from web/game.html in a Node vm with stubbed
// browser globals (document, localStorage, performance, matchMedia,
// requestAnimationFrame). No DOM is rendered and no browser is needed.
// The harness checks action-state identity: an AI action derived from a
// fixed aiGame must not change when the global live game G (or the global
// aiKind selector) is mutated, and a paired episode from one seed must
// replay identically.
//
// Usage: node tests/stage3_deeparcade_0_02_harness.js
// Exit code 0 when every check passes, 1 otherwise.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(ROOT, "web", "game.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
// The game logic is the second script block; the first is a policy stub.
const gameScript = scripts[1];
if (!gameScript || !gameScript.includes("function aiAction")) {
  throw new Error("game script block with aiAction not found in web/game.html");
}

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
  "\n;globalThis.__harness={Snake,aiAction,qAct,newGame,getG:()=>G," +
  "setG:(g)=>{G=g;},getAiKind:()=>aiKind,setAiKind:(k)=>{aiKind=k;}," +
  "getRecording:()=>recording,QPOLICY,STRAIGHT,LEFT,RIGHT};\n";
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

// 1. aiAction takes an explicit kind, not the global selector or G.
check("aiAction arity is 2 (game, kind)", H.aiAction.length === 2,
  "arity=" + H.aiAction.length);
check("aiAction source names an explicit kind parameter",
  H.aiAction.toString().includes("kind"));
if (H.aiAction.length !== 2) {
  // On the old signature every call below would dispatch on the global
  // selector (grandmaster MCTS per step), so stop here with a clear
  // failure instead of running slow stochastic episodes.
  console.log("1 check(s) failed: aiAction has no explicit kind parameter");
  process.exit(1);
}

// 2. Mutating global G and the global aiKind cannot change a fixed aiGame action.
const aiGame = new H.Snake(12, 12, 12345, "classic");
const state = aiGame.encodeState();
H.QPOLICY[state] = 2;
H.setAiKind("grandmaster");
const before = H.aiAction(aiGame, "rookie");
const other = new H.Snake(12, 12, 999, "maze");
other.body = [[5, 5], [5, 6], [5, 7]];
other.food = [0, 0];
other.heading = 3;
H.setG(other);
H.setAiKind("grandmaster");
const after = H.aiAction(aiGame, "rookie");
check("fixed aiGame rookie action invariant under G and aiKind mutation",
  before === 2 && after === 2, "before=" + before + " after=" + after);
check("fixed aiGame state itself untouched by the G mutation",
  aiGame.encodeState() === state);

// 3. Repeatable paired fixture: same seed, same policy, same outcome.
function runRookieEpisode(seed) {
  const game = new H.Snake(12, 12, seed, "classic");
  const actions = [];
  let guard = 0;
  while (!game.done && guard < 5000) {
    const a = H.aiAction(game, "rookie");
    actions.push(a);
    game.step(a);
    guard++;
  }
  return { score: game.score, steps: guard, done: game.done, actions };
}
const run1 = runRookieEpisode(12345);
const run2 = runRookieEpisode(12345);
check("AI episode terminates", run1.done && run1.steps <= 5000,
  "steps=" + run1.steps + " done=" + run1.done);
check("paired episode repeats exactly from the same seed",
  JSON.stringify(run1) === JSON.stringify(run2));
const runOtherSeed = runRookieEpisode(777);
check("AI episode terminates on a second seed",
  runOtherSeed.done && runOtherSeed.steps <= 5000);

// 4. Paired human/AI comparison on one shared seed (fixture only; the
// browser human Elo in localStorage is untouched by this comparison).
function runHumanFixture(seed) {
  const game = new H.Snake(12, 12, seed, "classic");
  let guard = 0;
  while (!game.done && guard < 5000) {
    game.step(H.STRAIGHT);
    guard++;
  }
  return game.score;
}
function resultLabel(hu, ai) {
  return hu > ai ? "Win" : hu === ai ? "Draw" : "Loss";
}
const huScore = runHumanFixture(12345);
const aiScore = run1.score;
console.log("paired fixture seed 12345: human=" + huScore + " ai=" + aiScore +
  " steps=" + run1.steps);
const label = resultLabel(huScore, aiScore);
check("paired human/AI result is a win, draw, or loss",
  label === "Win" || label === "Draw" || label === "Loss",
  "human=" + huScore + " ai=" + aiScore + " label=" + label);
check("paired human/AI result repeats from the same seed",
  resultLabel(runHumanFixture(12345), runRookieEpisode(12345).score) === label);
check("win/draw/loss labels follow the paired comparison rule",
  resultLabel(3, 1) === "Win" && resultLabel(2, 2) === "Draw" &&
  resultLabel(1, 3) === "Loss");

if (failures.length) {
  console.log(failures.length + " check(s) failed");
  process.exit(1);
}
console.log("harness: all checks passed");
