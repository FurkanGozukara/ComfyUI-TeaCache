import { app } from "../../../scripts/app.js";

const MASTER_WIDGET = "enable_speedup";
const OPTIMIZER_CLASS = "MiniMaxH3SpeedOptimizer";
const BACKEND_ORDER = Symbol("minimaxH3BackendWidgetOrder");
const BACKEND_DEFAULTS = Symbol("minimaxH3BackendWidgetDefaults");
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

function loadValuesInBackendOrder(node, data) {
    const named = data?.widgets_values_named;
    const positional = Array.isArray(data?.widgets_values) ? data.widgets_values : [];
    const widgets = backendOrderedWidgets(node);
    for (const [index, widget] of widgets.entries()) {
        if (named && Object.hasOwn(named, widget.name)) {
            widget.value = named[widget.name];
        } else if (index < positional.length) {
            widget.value = positional[index];
        }
    }

    // Older versions of this script let frontend 1.53+ save the values in display order,
    // which rotated them by one slot per reload (a combo value ends up in a number, ...).
    // That cannot be undone, so fall back to the defaults with the speedup switched off.
    const defaults = node[BACKEND_DEFAULTS];
    if (widgets.some((widget) => defaults.has(widget.name) && typeof widget.value !== typeof defaults.get(widget.name))) {
        for (const widget of widgets) {
            if (defaults.has(widget.name)) {
                widget.value = widget.name === MASTER_WIDGET ? false : defaults.get(widget.name);
            }
        }
        console.warn(`[MiniMaxH3Speed] node ${node.id}: saved widget values were scrambled by an older `
            + "version of this extension. They were reset to the defaults with the 4x speedup off; "
            + "reload the preset to get its tuned values back.");
    }
}

function promoteMasterSwitch(node) {
    if (!node.widgets) {
        return;
    }

    const index = node.widgets.findIndex((widget) => widget.name === MASTER_WIDGET);
    if (index > 0) {
        const [widget] = node.widgets.splice(index, 1);
        node.widgets.unshift(widget);
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
    // so translate the values in the node hooks instead of reordering node.widgets around it.
    // A subgraph host needs none of this: its values follow its input slots, not its widgets.
    if (node.comfyClass === OPTIMIZER_CLASS) {
        node[BACKEND_ORDER] = node.widgets.map((candidate) => candidate.name);
        node[BACKEND_DEFAULTS] = new Map(node.widgets.map((candidate) => [candidate.name, candidate.value]));

        const originalOnSerialize = node.onSerialize;
        node.onSerialize = function (data) {
            const result = originalOnSerialize?.apply(this, arguments);
            saveValuesInBackendOrder(this, data);
            return result;
        };

        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (data) {
            loadValuesInBackendOrder(this, data);
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
