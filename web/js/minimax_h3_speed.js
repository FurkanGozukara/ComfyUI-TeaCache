import { app } from "../../../scripts/app.js";

const MASTER_WIDGET = "enable_speedup";
const OPTIMIZER_CLASS = "MiniMaxH3SpeedOptimizer";
const BACKEND_ORDER = Symbol("minimaxH3BackendWidgetOrder");
const BACKEND_DEFAULTS = Symbol("minimaxH3BackendWidgetDefaults");
const REORDER_REFUSED = Symbol("minimaxH3WidgetReorderRefused");
const INSTALLED = Symbol("minimaxH3MasterSwitchInstalled");
const CALLBACK_INSTALLED = Symbol("minimaxH3MasterSwitchCallbackInstalled");

function isSpeedControlNode(node) {
    if (node.comfyClass === OPTIMIZER_CLASS) {
        return true;
    }

    // The stock I2V/T2V presets expose the optimizer switch on a generated subgraph
    // node, which does not consistently expose isSubgraphNode across frontend versions.
    // Requiring the paired threshold keeps this extension scoped to those H3 presets.
    return Boolean(
        node.widgets?.some((widget) => widget.name === MASTER_WIDGET)
        && node.widgets?.some((widget) => widget.name === "speed_threshold")
    );
}

function serializedWidgets(node) {
    return (node.widgets ?? []).filter((widget) => widget.serialize !== false);
}

// A sorted copy: node.widgets itself keeps the master switch first.
function backendOrderedWidgets(node) {
    const ranks = new Map(node[BACKEND_ORDER].map((name, index) => [name, index]));
    const rank = (widget) => ranks.get(widget.name) ?? Number.MAX_SAFE_INTEGER;
    return serializedWidgets(node).sort((left, right) => rank(left) - rank(right));
}

function saveValuesInBackendOrder(node, data) {
    const displayed = serializedWidgets(node);
    if (!Array.isArray(data?.widgets_values) || data.widgets_values.length !== displayed.length) {
        return;
    }

    const values = new Map(displayed.map((widget, index) => [widget, data.widgets_values[index]]));
    data.widgets_values = backendOrderedWidgets(node).map((widget) => values.get(widget));
}

// Which frontend saved a workflow, and whether it honoured onSerialize, is unknown when it is
// loaded again. Every layout the values can arrive in is therefore read as a candidate (one
// value per backend-ordered widget, undefined = not saved) and the first one whose values fit
// the widgets wins. Only when nothing fits are the values truly scrambled.
function loadValuesInBackendOrder(node, data) {
    const widgets = backendOrderedWidgets(node);
    const defaults = node[BACKEND_DEFAULTS];
    const fits = (values) => values.some((value) => value !== undefined)
        && widgets.every((widget, index) => {
            const value = values[index];
            if (value === undefined || !defaults.has(widget.name)) {
                return true;
            }
            const choices = widget.options?.values;
            return typeof value === typeof defaults.get(widget.name)
                && (!Array.isArray(choices) || typeof value !== "string" || choices.includes(value));
        });

    const positional = Array.isArray(data?.widgets_values) ? data.widgets_values : [];
    const named = data?.widgets_values_named;
    const master = widgets.filter((widget) => widget.name === MASTER_WIDGET);
    const displayed = [...master, ...widgets.filter((widget) => widget.name !== MASTER_WIDGET)];
    const candidates = [
        widgets.map((widget, index) => positional[index]),
        widgets.map((widget) => (named && Object.hasOwn(named, widget.name) ? named[widget.name] : undefined)),
        widgets.map((widget) => positional[displayed.indexOf(widget)]),
        widgets.map((widget) => widget.value),
    ];
    const values = candidates.find(fits);
    for (const [index, widget] of widgets.entries()) {
        if (values) {
            if (values[index] !== undefined) {
                widget.value = values[index];
            }
        } else if (defaults.has(widget.name)) {
            widget.value = widget.name === MASTER_WIDGET ? false : defaults.get(widget.name);
        }
    }

    // Older versions of this script let frontend 1.53+ save the values in display order and
    // restore them in backend order: one slot of rotation per reload, which cannot be undone.
    if (!values) {
        console.warn(`[MiniMaxH3Speed] node ${node.id}: saved widget values were scrambled by an older `
            + "version of this extension. They were reset to the defaults with the 4x speedup off; "
            + "reload the preset to get its tuned values back.");
    }
}

// Nothing in here may break loading or saving a workflow, whatever the frontend does.
function guarded(label, run) {
    try {
        return run();
    } catch (error) {
        console.warn(`[MiniMaxH3Speed] ${label} skipped:`, error);
        return undefined;
    }
}

function promoteMasterSwitch(node) {
    if (!node.widgets || node[REORDER_REFUSED]) {
        return;
    }

    // A frontend that does not allow reordering simply keeps the switch where it is. One
    // splice call, so a refusal can never leave the list without the widget.
    const index = node.widgets.findIndex((widget) => widget.name === MASTER_WIDGET);
    if (index > 0) {
        const reordered = [node.widgets[index], ...node.widgets.slice(0, index), ...node.widgets.slice(index + 1)];
        node[REORDER_REFUSED] = guarded("moving the master switch to the first row",
            () => node.widgets.splice(0, reordered.length, ...reordered)) === undefined;
    }
}

function drawMasterSwitchBorder(node, widget, ctx) {
    if (node.flags?.collapsed || !Number.isFinite(widget.last_y)) {
        return;
    }

    const enabled = Boolean(widget.value);
    const color = enabled ? "#22c55e" : "#f59e0b";
    const x = 7;
    const y = widget.last_y - 1;
    const width = Math.max(0, node.size[0] - 14);
    const height = (LiteGraph.NODE_WIDGET_HEIGHT || 20) + 2;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 4;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, 4);
    } else {
        ctx.rect(x, y, width, height);
    }
    ctx.stroke();
    ctx.restore();
}

function installMasterSwitch(node) {
    if (!isSpeedControlNode(node)) {
        return;
    }

    const widget = node.widgets?.find((candidate) => candidate.name === MASTER_WIDGET);
    if (!widget) {
        return;
    }

    widget.label = "4x speedup";
    if (!widget[CALLBACK_INSTALLED]) {
        widget[CALLBACK_INSTALLED] = true;
        const originalCallback = widget.callback;
        widget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            node.setDirtyCanvas?.(true, true);
            return result;
        };
    }

    if (node[INSTALLED]) {
        promoteMasterSwitch(node);
        node.setDirtyCanvas?.(true, true);
        return;
    }

    node[INSTALLED] = true;

    // Workflow widget values are positional while the master switch is displayed first.
    // Frontends differ in how they reach serialization (1.53+ never calls node.serialize()),
    // and ComfyUI and SwarmUI each choose their own frontend version, so the values are
    // translated in the node hooks instead of reordering node.widgets around internal methods.
    // A subgraph host needs none of this: its values follow its input slots, not its widgets.
    if (node.comfyClass === OPTIMIZER_CLASS) {
        node[BACKEND_ORDER] = node.widgets.map((candidate) => candidate.name);
        node[BACKEND_DEFAULTS] = new Map(node.widgets.map((candidate) => [candidate.name, candidate.value]));

        const originalOnSerialize = node.onSerialize;
        node.onSerialize = function (data) {
            const result = originalOnSerialize?.apply(this, arguments);
            guarded("saving the widget values in backend order", () => saveValuesInBackendOrder(this, data));
            return result;
        };

        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (data) {
            guarded("restoring the widget values", () => loadValuesInBackendOrder(this, data));
            return originalOnConfigure?.apply(this, arguments);
        };
    }

    const originalDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        const result = originalDrawForeground?.apply(this, arguments);
        const activeWidget = this.widgets?.find((candidate) => candidate.name === MASTER_WIDGET);
        if (activeWidget) {
            drawMasterSwitchBorder(this, activeWidget, ctx);
        }
        return result;
    };

    promoteMasterSwitch(node);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "TeaCache.MiniMaxH3MasterSwitch",
    nodeCreated(node) {
        installMasterSwitch(node);
    },
    loadedGraphNode(node) {
        setTimeout(() => installMasterSwitch(node), 0);
    },
    afterConfigureGraph() {
        setTimeout(() => {
            for (const node of app.graph?._nodes ?? []) {
                installMasterSwitch(node);
            }
        }, 0);
    },
});
