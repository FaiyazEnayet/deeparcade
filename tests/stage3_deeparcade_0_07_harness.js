// Stage 3 deeparcade P0-07: isolated nonvisual harness for web/game.html.
//
// Loads the real game script from web/game.html in a Node vm with stubbed
// browser globals (document, localStorage, performance, matchMedia,
// requestAnimationFrame). No DOM is rendered and no browser is needed.
// The harness replays the deterministic transition fixtures in
// tests/fixtures/stage3_deeparcade_0_07_fixtures.json (identical actions,
// scripted food placements) against the browser Snake core and compares
// body, head, heading, food, score, steps, reward, terminal flags and the
// 32-state encoding after every step with the Python specification.
// It also checks terminal behavior (a full-board win ends the game, step
// and policy observations stay consistent once done), the 12x12 DQN
// encoding, maze and blitz smoke behavior, and classic human input
// routing through the real tick path.
//
// Usage: node tests/stage3_deeparcade_0_07_harness.js
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
if (!gameScript || !gameScript.includes("class Snake")) {
  throw new Error("game script block with the Snake class not found in web/game.html");
}
const fixtures = JSON.parse(fs.readFileSync(
  path.join(__dirname, "fixtures", "stage3_deeparcade_0_07_fixtures.json"), "utf8"));

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
  "\n;globalThis.__harness={Snake,RNG,mazeWalls,qAct,aiAction,newGame,tick," +
  "STRAIGHT,LEFT,RIGHT,getG:()=>G,pushInput:(a)=>{inputQueue.push(a);}," +
  "queueLen:()=>inputQueue.length};\n";
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

// Replay one fixture scenario against the browser core with the start
// state injected and food placement scripted by the fixture queue.
function replayJS(scn) {
  const g = new H.Snake(scn.width, scn.height, 1, "classic");
  const queue = scn.food_script.map((c) => c.slice());
  g._placeFood = function () {
    if (queue.length) {
      const cell = queue.shift();
      if (this.body.some((p) => p[0] === cell[0] && p[1] === cell[1])) {
        throw new Error("scripted food placed on the body in " + scn.name);
      }
      this.food = cell;
    } else {
      this.food = null;
    }
  };
  g.body = scn.start.body.map((c) => c.slice());
  g.heading = scn.start.heading;
  g.score = scn.start.score;
  g.steps = scn.start.steps;
  g.done = false;
  g.won = false;
  g._placeFood();
  const records = [snapJS(g, null)];
  for (const a of scn.actions) {
    const r = g.step(a);
    records.push(snapJS(g, r.reward));
    if (g.done) break;
  }
  return { game: g, records };
}

function snapJS(g, reward) {
  return {
    body: g.body.map((c) => c.slice()),
    heading: g.heading,
    food: g.food ? g.food.slice() : null,
    score: g.score,
    steps: g.steps,
    reward,
    done: g.done,
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
  if (exp.state !== got.state) return "state";
  return null;
}

// 1. Fixture replays: browser transitions equal the Python specification.
for (const scn of fixtures.scenarios) {
  let out;
  try {
    out = replayJS(scn);
  } catch (e) {
    check("fixture " + scn.name + " replays without throwing", false,
      e.constructor.name + ": " + e.message);
    continue;
  }
  check("fixture " + scn.name + " replays without throwing", true);
  const { game, records } = out;
  let bad = null;
  if (records.length !== scn.records.length) {
    bad = "record count " + records.length + " != " + scn.records.length;
  } else {
    for (let i = 0; i < records.length; i++) {
      const d = recordDiff(scn.records[i], records[i]);
      if (d) { bad = "record " + i + " field " + d; break; }
    }
  }
  check("fixture " + scn.name + " matches the Python transitions step for step",
    bad === null, bad || undefined);
  // Head is the last body cell in both cores; pin it explicitly.
  const lastExp = scn.records[scn.records.length - 1];
  const lastGot = records[records.length - 1];
  check("fixture " + scn.name + " final head agrees",
    JSON.stringify(lastExp.body[lastExp.body.length - 1]) ===
    JSON.stringify(lastGot.body[lastGot.body.length - 1]));
  // Terminal behavior: the game must be done exactly when the fixture
  // ends, a win must be flagged as a win, and terminal observations must
  // stay consistent (encodeState repeatable, step refused) once done.
  if (lastExp.done) {
    check("fixture " + scn.name + " ends done", game.done === true);
    check("fixture " + scn.name + " win flag matches the Python outcome",
      (game.won === true) === (lastExp.reason === "win"),
      "won=" + game.won + " reason=" + lastExp.reason);
    let s1, s2, threw = false;
    try { s1 = game.encodeState(); s2 = game.encodeState(); } catch (e) { threw = true; }
    check("fixture " + scn.name + " terminal observation is consistent",
      !threw && s1 === lastExp.state && s2 === lastExp.state,
      threw ? "encodeState threw on a done game" : "state=" + s1);
    let stepThrew = false;
    try { game.step(H.STRAIGHT); } catch (e) { stepThrew = true; }
    check("fixture " + scn.name + " refuses to step once done", stepThrew);
  }
  // DQN encoding checks pinned by the fixture (12x12 scenarios).
  if (scn.dqn_before) {
    const fresh = replayJS({ ...scn, actions: [] }).game;
    check("fixture " + scn.name + " encodeDQN matches Python before the step",
      JSON.stringify(Array.from(fresh.encodeDQN())) ===
      JSON.stringify(scn.dqn_before));
  }
  if (scn.dqn_after) {
    let vec = null, threw = false;
    try { vec = Array.from(game.encodeDQN()); } catch (e) { threw = true; }
    check("fixture " + scn.name + " encodeDQN stays defined on the terminal win",
      !threw && JSON.stringify(vec) === JSON.stringify(scn.dqn_after),
      threw ? "encodeDQN threw on a done game" : undefined);
  }
}

// 2. A policy must never be invoked on a finished game by the game loop:
// drive the real tick path to a death, then confirm tick stops stepping.
H.newGame("classic", "human", undefined);
const G = H.getG();
G.food = [0, 0]; // keep the scripted drive free of eats
let guard = 0;
while (!G.done && guard < 50) { H.tick(); guard++; }
check("human tick drive reaches a terminal state", G.done === true,
  "steps=" + G.steps);
const stepsAtDone = G.steps;
H.tick(); H.tick();
check("tick does not step or throw once the game is done",
  G.steps === stepsAtDone, "steps " + stepsAtDone + " -> " + G.steps);

// 3. Classic human input routing: queued turns apply on the next ticks.
H.newGame("classic", "human", undefined);
const G2 = H.getG();
G2.food = [0, 0];
const h0 = G2.heading;
H.pushInput(H.LEFT);
H.tick();
check("queued LEFT turn applies on the next tick",
  G2.heading === (h0 + 3) % 4 && G2.steps === 1 && !G2.done,
  "heading " + h0 + " -> " + G2.heading);
H.tick();
check("empty input queue continues straight",
  G2.heading === (h0 + 3) % 4 && G2.steps === 2 && !G2.done,
  "heading=" + G2.heading + " steps=" + G2.steps);

// 4. Maze smoke: walls exist, food never lands on a wall or the body, and
// a wall cell is lethal and reads as danger in the state encoding.
for (let seed = 1; seed <= 20; seed++) {
  const m = new H.Snake(12, 12, seed, "maze");
  if (m.walls.size !== 12) { check("maze walls are built", false, "seed " + seed); break; }
  if (m.food && (m.walls.has(m.food[0] + "," + m.food[1]) ||
      m.body.some((p) => p[0] === m.food[0] && p[1] === m.food[1]))) {
    check("maze food placement avoids walls and body", false, "seed " + seed);
    break;
  }
  if (seed === 20) check("maze food placement avoids walls and body", true);
}
const mz = new H.Snake(12, 12, 3, "maze");
mz.body = [[3, 0], [3, 1], [3, 2]]; // head (3,2), wall block at (3,3)
mz.heading = 1; mz.food = [0, 0]; mz.done = false;
const mazeState = mz.encodeState();
check("maze wall ahead reads as danger straight",
  Math.floor(mazeState / 16) === 1, "state=" + mazeState);
const mzr = mz.step(H.STRAIGHT);
check("stepping into a maze wall is lethal",
  mzr.done === true && mzr.reward === -10 && mz.done === true);

// 5. Blitz smoke: eating speeds the game up, with a floor.
const bz = new H.Snake(12, 12, 1, "blitz");
check("blitz starts at 130 ms ticks", bz.tickMs === 130, "tickMs=" + bz.tickMs);
function blitzEat(game) {
  game.body = [[6, 4], [6, 5], [6, 6]];
  game.heading = 1; game.food = [6, 7]; game.done = false;
  return game.step(H.STRAIGHT);
}
blitzEat(bz);
check("blitz eat speeds the tick up by 1.5 ms", bz.tickMs === 128.5,
  "tickMs=" + bz.tickMs);
for (let i = 0; i < 60; i++) blitzEat(bz);
check("blitz tick floor holds at 60 ms", bz.tickMs === 60,
  "tickMs=" + bz.tickMs);

if (failures.length) {
  console.log(failures.length + " check(s) failed");
  process.exit(1);
}
console.log("harness: all checks passed");
