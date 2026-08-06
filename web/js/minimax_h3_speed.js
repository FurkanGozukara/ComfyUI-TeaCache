import { app } from "../../../scripts/app.js";

const MASTER_WIDGET = "enable_speedup";
const OPTIMIZER_CLASS = "MiniMaxH3SpeedOptimizer";
const BACKEND_ORDER = Symbol("minimaxH3BackendWidgetOrder");
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

function restoreBackendOrder(node) {
    if (!node.widgets || !node[BACKEND_ORDER]) {
        return;
    }

    const ranks = new Map(node[BACKEND_ORDER].map((name, index) => [name, index]));
    const stableIndex = new Map(node.widgets.map((widget, index) => [widget, index]));
    node.widgets.sort((left, right) => {
        const leftRank = ranks.get(left.name) ?? Number.MAX_SAFE_INTEGER;
        const rightRank = ranks.get(right.name) ?? Number.MAX_SAFE_INTEGER;
        return leftRank - rightRank || stableIndex.get(left) - stableIndex.get(right);
    });
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
    node[BACKEND_ORDER] = node.widgets.map((candidate) => candidate.name);

    // Workflow widget values are positional. Restore the backend order only while
    // configuring/serializing, then return the master switch to the first visible row.
    const originalConfigure = node.configure;
    if (typeof originalConfigure === "function") {
        node.configure = function (...args) {
            restoreBackendOrder(this);
            try {
                return originalConfigure.apply(this, args);
            } finally {
                promoteMasterSwitch(this);
            }
        };
    }

    const originalSerialize = node.serialize;
    if (typeof originalSerialize === "function") {
        node.serialize = function (...args) {
            restoreBackendOrder(this);
            try {
                return originalSerialize.apply(this, args);
            } finally {
                promoteMasterSwitch(this);
            }
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
