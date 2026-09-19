// Run with: node --test tests/minimax_h3_speed_widget_order.test.mjs
//
// web/js/minimax_h3_speed.js displays the enable_speedup master switch first, while workflow
// widgets_values are positional in backend (python INPUT_TYPES) order. Frontend 1.53+ serializes
// graphs through node.serializeFromStoreState() and never calls node.serialize(), so wrapping
// serialize/configure saved the values in display order and rotated them by one slot per
// reload (six reloads put fbc_cache_device "gpu" into fbc_end_percent). ComfyUI and SwarmUI
// each pick their own frontend version, so every save path modelled here has to stay correct,
// including frontends that do not exist yet.

import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const DEFS = [
    ["first_block_cache", true], ["fbc_threshold", 0.08], ["fbc_start_percent", 0.15], ["fbc_end_percent", 0.95],
    ["fbc_max_consecutive", 3], ["sparse_attention", "auto", ["auto", "enabled", "disabled"]],
    ["sparse_dense_steps_pct", 0.2], ["sparse_dense_layers", 2], ["sparse_tau", 1.0], ["sparse_min_video_rows", 4096],
    ["fbc_cache_device", "gpu", ["gpu", "cpu"]], ["verbose", true], ["enable_speedup", true],
    ["sparse_backend", "auto", ["auto", "comfy_kitchen", "vendored"]], ["sparse_extra_tokens", 256],
    ["sparse_dense_last_steps", 1],
];
const NAMES = DEFS.map(([name]) => name);
const DEFAULTS = Object.fromEntries(DEFS.map(([name, value]) => [name, value]));
const DEFAULTS_OFF = { ...DEFAULTS, enable_speedup: false };

// Node 401 of the stock "References To Video" preset: saved before the last three widgets existed.
const PRESET_NAMED = Object.fromEntries(NAMES.slice(0, 13).map((name) => [name, DEFAULTS[name]]));
PRESET_NAMED.sparse_tau = 1;
PRESET_NAMED.enable_speedup = false;
const PRESET = { widgets_values: NAMES.slice(0, 13).map((name) => PRESET_NAMED[name]), widgets_values_named: PRESET_NAMED };
const EXPECTED = { ...DEFAULTS, ...PRESET_NAMED };
const EDITED = { ...EXPECTED, enable_speedup: true, fbc_threshold: 0.12, fbc_cache_device: "cpu" };

async function loadExtension() {
    let extension = null;
    globalThis.__minimaxH3App = { registerExtension: (candidate) => { extension = candidate; }, graph: { _nodes: [] } };
    const source = fs.readFileSync(new URL("../web/js/minimax_h3_speed.js", import.meta.url), "utf8")
        .replace(/^import .*app\.js";\s*$/m, "const app = globalThis.__minimaxH3App;");
    await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
    return extension;
}

const EXTENSION = await loadExtension();

// kind: "legacy"  graph saves call node.serialize()                       (frontend <= 1.52)
//       "store"   graph saves call node.serializeFromStoreState()          (frontend 1.53+)
//       "hostile" a future frontend: no onSerialize, no widgets_values_named
//       "frozen"  a future frontend whose widget list cannot be reordered
function makeFrontend(kind, namedValuesRestore = false) {
    class Node {
        comfyClass = "MiniMaxH3SpeedOptimizer";
        id = 401;

        constructor() {
            const widgets = DEFS.map(([name, value, values]) => ({ name, value, options: values ? { values } : {} }));
            this.widgets = kind === "frozen" ? Object.freeze(widgets) : widgets;
        }

        serializeFromStoreState() {
            const data = { widgets_values: this.widgets.map((widget) => widget.value) };
            if (kind !== "hostile") {
                data.widgets_values_named = Object.fromEntries(this.widgets.map((widget) => [widget.name, widget.value]));
                this.onSerialize?.(data);
            }
            return data;
        }

        serialize() {
            return this.serializeFromStoreState();
        }

        configure(data) {
            for (const [index, widget] of this.widgets.entries()) {
                if (namedValuesRestore && data.widgets_values_named) {
                    if (Object.hasOwn(data.widgets_values_named, widget.name)) widget.value = data.widgets_values_named[widget.name];
                } else if (Array.isArray(data.widgets_values) && index < data.widgets_values.length) {
                    widget.value = data.widgets_values[index];
                }
            }
            this.onConfigure?.(kind === "legacy" ? data : { ...data });
        }
    }

    const load = (saved) => {
        const node = new Node();
        EXTENSION.nodeCreated(node);
        node.configure(structuredClone(saved));
        return node;
    };
    const save = (node) => structuredClone(kind === "legacy" ? node.serialize() : node.serializeFromStoreState());
    return { load, save };
}

const state = (node) => Object.fromEntries(node.widgets.map((widget) => [widget.name, widget.value]));
const edit = (node, values) => Object.entries(values).forEach(([name, value]) => {
    node.widgets.find((widget) => widget.name === name).value = value;
});

// What the old script persisted on frontend 1.53+ after `reloads` reloads: display-order values.
function legacyDraft(reloads) {
    let values = NAMES.map((name) => EXPECTED[name]);
    for (let cycle = 0; cycle < reloads; cycle++) {
        values = [values[12], ...values.slice(0, 12), ...values.slice(13)];
    }
    const named = Object.fromEntries(NAMES.map((name, index) => [name, values[index]]));
    return { widgets_values: [values[12], ...values.slice(0, 12), ...values.slice(13)], widgets_values_named: named };
}

function quietly(run) {
    const original = console.warn;
    const warnings = [];
    console.warn = (...message) => warnings.push(message.join(" "));
    try {
        const node = run();
        // A frontend that refuses the reordering is reported once per node; that is not a reset.
        return { node, warnings: warnings.filter((message) => message.includes("scrambled")) };
    } finally {
        console.warn = original;
    }
}

for (const kind of ["legacy", "store", "hostile", "frozen"]) {
    for (const namedValuesRestore of [false, true]) {
        const frontend = makeFrontend(kind, namedValuesRestore);
        const label = `${kind} frontend, namedValuesRestore=${namedValuesRestore}`;

        test(`${label}: the stock preset and later edits survive every reload`, () => {
            const { node: last, warnings } = quietly(() => {
                let node = frontend.load(PRESET);
                assert.deepEqual(state(node), EXPECTED);
                edit(node, { enable_speedup: true, fbc_threshold: 0.12, fbc_cache_device: "cpu" });
                for (let cycle = 0; cycle < 25; cycle++) {
                    node = frontend.load(frontend.save(node));
                    assert.deepEqual(state(node), EDITED, `reload ${cycle + 1}`);
                }
                return node;
            });
            assert.deepEqual(warnings, []);
            assert.equal(last.widgets[0].name, kind === "frozen" ? "first_block_cache" : "enable_speedup");
            if (kind === "legacy" || kind === "store") {
                assert.deepEqual(frontend.save(last).widgets_values, NAMES.map((name) => EDITED[name]));
            }
        });

        test(`${label}: saves without named values, copies and unknown formats keep their values`, () => {
            assert.deepEqual(state(frontend.load({ widgets_values: PRESET.widgets_values })), EXPECTED);
            const node = frontend.load(PRESET);
            assert.deepEqual(state(frontend.load(structuredClone(node.serialize()))), EXPECTED);
            const { node: unknown, warnings } = quietly(() => frontend.load({ widgets_values: { fbc_threshold: 0.5 } }));
            assert.deepEqual(state(unknown), DEFAULTS);
            assert.deepEqual(warnings, []);
        });

        test(`${label}: a display-order draft of the old script is recovered`, () => {
            const { node, warnings } = quietly(() => frontend.load(legacyDraft(0)));
            assert.deepEqual(state(node), EXPECTED);
            assert.deepEqual(warnings, []);
            const bare = quietly(() => frontend.load({ widgets_values: legacyDraft(0).widgets_values }));
            assert.deepEqual(state(bare.node), EXPECTED);
            assert.deepEqual(bare.warnings, []);
        });

        test(`${label}: every rotated draft of the old script resets to the defaults with the speedup off`, () => {
            assert.equal(legacyDraft(6).widgets_values_named.fbc_end_percent, "gpu");
            // The twelfth rotation saved the display order of a state whose backend reading is
            // the original again, so that one draft is recovered instead of reset.
            const recovered = quietly(() => frontend.load(legacyDraft(12)));
            assert.deepEqual(state(recovered.node), EXPECTED);
            assert.deepEqual(recovered.warnings, []);
            for (let reloads = 1; reloads <= 11; reloads++) {
                const { node, warnings } = quietly(() => frontend.load(legacyDraft(reloads)));
                assert.deepEqual(state(node), DEFAULTS_OFF, `rotation ${reloads}`);
                assert.equal(warnings.length, 1, `rotation ${reloads}`);
                const again = quietly(() => frontend.load(frontend.save(node)));
                assert.deepEqual(state(again.node), DEFAULTS_OFF);
                assert.deepEqual(again.warnings, []);
            }
        });
    }
}

test("positional values win over stale named values, as in the frontend itself", () => {
    const stale = { ...PRESET, widgets_values_named: { ...PRESET_NAMED, fbc_threshold: 0.5 } };
    assert.equal(state(makeFrontend("store").load(stale)).fbc_threshold, 0.08);
});

test("a subgraph host is only reordered visually: its values follow its input slots", () => {
    const host = { comfyClass: "subgraph", id: 7, widgets: [{ name: "speed_threshold", value: 0.08 }, { name: "enable_speedup", value: true }] };
    EXTENSION.nodeCreated(host);
    assert.equal(host.widgets[0].name, "enable_speedup");
    assert.equal(host.onSerialize, undefined);
    assert.equal(host.onConfigure, undefined);
});
