// Run with: node --test tests/minimax_h3_speed_widget_order.test.mjs
//
// web/js/minimax_h3_speed.js displays the enable_speedup master switch first, while workflow
// widgets_values are positional in backend (python INPUT_TYPES) order. Frontend 1.53+ serializes
// graphs through node.serializeFromStoreState() and never calls node.serialize(), so wrapping
// serialize/configure saved the values in display order and rotated them by one slot per
// reload (six reloads put fbc_cache_device "gpu" into fbc_end_percent). Both save paths are
// modelled here; the script has to stay correct on each of them.

import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const DEFS = [
    ["first_block_cache", true], ["fbc_threshold", 0.08], ["fbc_start_percent", 0.15], ["fbc_end_percent", 0.95],
    ["fbc_max_consecutive", 3], ["sparse_attention", "auto"], ["sparse_dense_steps_pct", 0.2], ["sparse_dense_layers", 2],
    ["sparse_tau", 1.0], ["sparse_min_video_rows", 4096], ["fbc_cache_device", "gpu"], ["verbose", true],
    ["enable_speedup", true], ["sparse_backend", "auto"], ["sparse_extra_tokens", 256], ["sparse_dense_last_steps", 1],
];
const NAMES = DEFS.map(([name]) => name);
const DEFAULTS_OFF = { ...Object.fromEntries(DEFS), enable_speedup: false };

// Node 401 of the stock "References To Video" preset: saved before the last three widgets existed.
const PRESET_NAMED = { ...Object.fromEntries(DEFS.slice(0, 13)), sparse_tau: 1, enable_speedup: false };
const PRESET = { widgets_values: NAMES.slice(0, 13).map((name) => PRESET_NAMED[name]), widgets_values_named: PRESET_NAMED };
const EXPECTED = { ...Object.fromEntries(DEFS), ...PRESET_NAMED };

async function loadExtension() {
    let extension = null;
    globalThis.__minimaxH3App = { registerExtension: (candidate) => { extension = candidate; }, graph: { _nodes: [] } };
    const source = fs.readFileSync(new URL("../web/js/minimax_h3_speed.js", import.meta.url), "utf8")
        .replace(/^import .*app\.js";\s*$/m, "const app = globalThis.__minimaxH3App;");
    await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
    return extension;
}

const EXTENSION = await loadExtension();

function makeFrontend(kind, namedValuesRestore) {
    class Node {
        comfyClass = "MiniMaxH3SpeedOptimizer";
        id = 401;
        widgets = DEFS.map(([name, value]) => ({ name, value }));

        serializeFromStoreState() {
            const data = { widgets_values: [], widgets_values_named: {} };
            for (const widget of this.widgets) {
                data.widgets_values.push(widget.value);
                data.widgets_values_named[widget.name] = widget.value;
            }
            this.onSerialize?.(data);
            return data;
        }

        serialize() {
            return this.serializeFromStoreState();
        }

        configure(data) {
            for (const [index, widget] of this.widgets.entries()) {
                if (namedValuesRestore && data.widgets_values_named) {
                    if (Object.hasOwn(data.widgets_values_named, widget.name)) widget.value = data.widgets_values_named[widget.name];
                } else if (index < (data.widgets_values?.length ?? 0)) {
                    widget.value = data.widgets_values[index];
                }
            }
            this.onConfigure?.(kind === "store" ? { ...data } : data);
        }
    }

    const load = (saved) => {
        const node = new Node();
        EXTENSION.nodeCreated(node);
        node.configure(structuredClone(saved));
        return node;
    };
    // Legacy graphs call node.serialize(); 1.53+ calls node.serializeFromStoreState() directly.
    const save = (node) => structuredClone(kind === "store" ? node.serializeFromStoreState() : node.serialize());
    return { load, save };
}

const state = (node) => Object.fromEntries(node.widgets.map((widget) => [widget.name, widget.value]));

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
    console.warn = (message) => warnings.push(message);
    try {
        return { node: run(), warnings };
    } finally {
        console.warn = original;
    }
}

for (const kind of ["legacy", "store"]) {
    for (const namedValuesRestore of [false, true]) {
        const frontend = makeFrontend(kind, namedValuesRestore);
        const label = `${kind} frontend, namedValuesRestore=${namedValuesRestore}`;

        test(`${label}: reloads never move a value and the saved array stays in backend order`, () => {
            let node = frontend.load(PRESET);
            for (let cycle = 0; cycle < 25; cycle++) {
                assert.deepEqual(state(node), EXPECTED);
                assert.equal(node.widgets[0].name, "enable_speedup");
                assert.deepEqual(frontend.save(node).widgets_values, NAMES.map((name) => EXPECTED[name]));
                node = frontend.load(frontend.save(node));
            }
        });

        test(`${label}: edits, copies and saves without named values keep their values`, () => {
            let node = frontend.load({ widgets_values: PRESET.widgets_values });
            assert.deepEqual(state(node), EXPECTED);
            node.widgets.find((widget) => widget.name === "enable_speedup").value = true;
            node.widgets.find((widget) => widget.name === "fbc_cache_device").value = "cpu";
            node = frontend.load(frontend.save(frontend.load(node.serialize())));
            assert.deepEqual(state(node), { ...EXPECTED, enable_speedup: true, fbc_cache_device: "cpu" });
        });

        test(`${label}: a display-order draft of the old script is read through its named values`, () => {
            const { node, warnings } = quietly(() => frontend.load(legacyDraft(0)));
            assert.deepEqual(state(node), EXPECTED);
            assert.equal(warnings.length, 0);
        });

        test(`${label}: every rotated draft of the old script resets to the defaults with the speedup off`, () => {
            assert.equal(legacyDraft(6).widgets_values_named.fbc_end_percent, "gpu");
            for (let reloads = 1; reloads <= 12; reloads++) {
                const { node, warnings } = quietly(() => frontend.load(legacyDraft(reloads)));
                assert.deepEqual(state(node), DEFAULTS_OFF, `rotation ${reloads}`);
                assert.equal(warnings.length, 1, `rotation ${reloads}`);
                assert.deepEqual(state(frontend.load(frontend.save(node))), DEFAULTS_OFF);
            }
        });
    }
}

test("a subgraph host is only reordered visually: its values follow its input slots", () => {
    const host = { comfyClass: "subgraph", id: 7, widgets: [{ name: "speed_threshold", value: 0.08 }, { name: "enable_speedup", value: true }] };
    EXTENSION.nodeCreated(host);
    assert.equal(host.widgets[0].name, "enable_speedup");
    assert.equal(host.onSerialize, undefined);
    assert.equal(host.onConfigure, undefined);
});
