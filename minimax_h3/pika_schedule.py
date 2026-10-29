"""Registers the Pika H3 Flex 6-step schedule as a ComfyUI scheduler named "pika_h3_flex".

Pika H3 Flex (Pika Labs' MiniMax H3 adapters) is distilled for one fixed sigma list, which Comfy-Org's official
templates feed through ManualSigmas. Frontends that only offer a scheduler list (SwarmUI's sampler) can pick it
by name; the values are the official list, so both routes give the same sigmas. Under the default H3 video shift
of 12 they are the base timesteps 1, 0.75, 0.5, 0.25, 0.1 and 0.04.
"""

import logging

import torch
import comfy.samplers

PIKA_H3_FLEX_SCHEDULER = "pika_h3_flex"
PIKA_H3_FLEX_SIGMAS = (1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0)


def pika_h3_flex_scheduler(model_sampling, steps):
    if int(steps) != len(PIKA_H3_FLEX_SIGMAS) - 1:
        logging.warning(
            "[pika_h3_flex] Pika H3 Flex is distilled for %d steps; using its %d-step schedule instead of %s steps.",
            len(PIKA_H3_FLEX_SIGMAS) - 1, len(PIKA_H3_FLEX_SIGMAS) - 1, steps,
        )
    return torch.FloatTensor(PIKA_H3_FLEX_SIGMAS)


if PIKA_H3_FLEX_SCHEDULER not in comfy.samplers.SCHEDULER_HANDLERS:
    comfy.samplers.SCHEDULER_HANDLERS[PIKA_H3_FLEX_SCHEDULER] = comfy.samplers.SchedulerHandler(pika_h3_flex_scheduler)
    comfy.samplers.SCHEDULER_NAMES.append(PIKA_H3_FLEX_SCHEDULER)
