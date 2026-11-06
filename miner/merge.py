"""Merged v5 (Claude research, round 2 loops 7+8): loop-6 pipeline + a 3rd post refine + a per-image render cache.
Loop 6 docstring follows. Merged v4 + concurrent chains (Claude research, round 2 loop 6): sedge-wren backbone + newvictor's side-view rewrite feedback.
Evidence: round_2/loop_5/idea.md.

Backbone: Olauge-Secret/sedge-wren@fc79ce3 FLOW="lumen" (miner/lumen.py): Astra-medium and Sol-medium chains, each
draft repaired, then rewritten Sol-high -> Astra-high; Astra-low pick; Sol-high and Astra-high post refines; finish
(wrap + flat-object tilt). Same models, efforts, prompts, pick, request ids and budget guard as lumen.py.

Merged from SmoothTight/newvictor@0f4bc93 (miner404/pipeline.py SIDES and prompts.REWRITE item 5): every rewrite turn
(chain rewrites and post refines) additionally shows the current module's SIDES sheet: right, back and left at eye level
and the top (theta 90/180/270/0, phi 0/0/0/-90), 256 px each on white with neutral light, as one 2x2 image, with
newvictor's instruction to keep every part recognizable and complete any back or side that is missing, flat or hollow.
The front views, refine text and code block are sedge's, unchanged and in the same order.

Robustness (not in sedge): a dropped connection (APIConnectionError) is resent with the SAME request id up to
TRANSPORT_RETRIES times instead of failing the whole image.
"""

from __future__ import annotations

import asyncio
import hashlib

import openai
from loguru import logger
from witness_client import AsyncWitnessClient, RenderOptions

from miner import images, lumen, pipeline, rewrite
from miner.rewrite import Calls, check, code_of, finish, first_module, image_part, refine_content, text_part

SIDES = RenderOptions(
    thetas=(90.0, 180.0, 270.0, 0.0),
    phis=(0.0, 0.0, 0.0, -90.0),
    img_size=256,
    cam_radius=2.0,
    cam_fov_deg=49.1,
    gap=images.GRID_GAP,
    bg_color="ffffff",
    lighting="neutral",
)
SIDES_LABEL = (
    "The same module seen from the other sides (one 2x2 image: top-left from the right, top-right from the back, "
    "bottom-left from the left, bottom-right from the top). Every part must stay recognizable from these sides too: "
    "complete any back or side that is missing, flat or hollow, without changing what already matches the photo in "
    "the front views:"
)
TRANSPORT_RETRIES, TRANSPORT_WAITS = 3, (5, 15, 45)
CHAIN_K0, POST_K0 = 40, 80
# Each chain's budget guard: own spend + this reserve (for the other chain and the post stage) + next worst case <= 5M.
# Loops 7/8: max chain spend 2.10M, chains combined 2.83M, so neither this nor the earlier shared guard ever bound.
CHAIN_RESERVE_CREDITS, CHAIN_RESERVE_CALLS = 1_000_000, 20
# sedge's post refine text already says "step k of 3" (lumen.POST_STEPS = 3) but runs 2 steps; v5 runs the 3rd:
POST_STAGES = (*lumen.POST_STAGES, (pipeline.MODEL, 3, ""))  # Sol high, "REFINEMENT STEP 3 of 3", with the sides sheet


class CachedWitness:
    """The witness with a per-image render cache: the renderer is deterministic, so an identical (source, options)
    pair is rendered once. Failures are not cached. Everything else passes through."""

    def __init__(self, witness: AsyncWitnessClient) -> None:
        self._w, self._cache, self.hits = witness, {}, 0

    async def render(self, source: str, options: RenderOptions = None):  # type: ignore[assignment]
        key = (hashlib.sha256(source.encode()).hexdigest(), repr(options))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = await (self._w.render(source) if options is None else self._w.render(source, options))
        self._cache[key] = out
        return out

    def __getattr__(self, name):
        return getattr(self._w, name)  # Calls.k bases: chain c uses k from 40*c (ids 4001+ for chain 2), pick/post from 80


class Spend:
    """One image's spend, shared by the concurrent chains' Calls (sedge's guard reads it before every call)."""

    def __init__(self) -> None:
        self.credits, self.calls = 0.0, 0


class RCalls(Calls):
    """sedge's Calls, resending a dropped connection with the same request ids. With `spend`, its credits/calls are
    the image's shared totals and its request ids start at k0 * 100, so concurrent chains never share an id."""

    def __init__(self, witness, spend: "Spend | None" = None, k0: int = 0) -> None:
        super().__init__(witness)
        self._spend = spend
        self.k = k0

    @property
    def credits(self) -> float:  # type: ignore[override]
        return self._spend.credits if getattr(self, "_spend", None) else self.__dict__.get("_credits", 0.0)

    @credits.setter
    def credits(self, v: float) -> None:
        if getattr(self, "_spend", None):
            self._spend.credits = v
        else:
            self.__dict__["_credits"] = v

    @property
    def calls(self) -> int:  # type: ignore[override]
        return self._spend.calls if getattr(self, "_spend", None) else self.__dict__.get("_calls", 0)

    @calls.setter
    def calls(self, v: int) -> None:
        if getattr(self, "_spend", None):
            self._spend.calls = v
        else:
            self.__dict__["_calls"] = v

    async def ask(self, content, effort, model=pipeline.ASTRA, instructions=rewrite.P.SYSTEM, max_output=None):  # type: ignore[override]
        k = self.k
        for attempt in range(TRANSPORT_RETRIES + 1):
            self.k = k
            try:
                return await super().ask(content, effort, model, instructions, max_output)
            except openai.APIConnectionError as e:
                if attempt == TRANSPORT_RETRIES:
                    raise pipeline.ModelFailure(f"call {k}: transport failed {attempt + 1}x ({e})") from e
                logger.warning("call {}: transport failure ({}), resending in {}s", k, str(e)[:80], TRANSPORT_WAITS[attempt])
                await asyncio.sleep(TRANSPORT_WAITS[attempt])
        raise AssertionError("unreachable")


async def refine_with_sides(step: int, steps: int, photo: dict[str, str], grid: bytes, module: str,
                            witness: AsyncWitnessClient) -> list[dict[str, str]]:
    """sedge's refine turn + the SIDES sheet before the code block (the code stays last, as in sedge)."""
    content = refine_content(step, steps, photo, grid, module)
    sides, _ = await check(module, witness, SIDES)
    if sides is None:  # rendered fine from the front a moment ago: a renderer flake, so sedge's turn as it is
        logger.warning("sides render failed, refining from the front views only")
        return content
    return [*content[:-1], text_part(SIDES_LABEL), image_part(images.sheet_data_url(sides, SIDES.img_size)), content[-1]]


async def chain(calls: Calls, photo: dict[str, str], model: str, effort: str) -> tuple[str, bool]:
    """lumen.chain with the sides sheet in each rewrite."""
    best, grid = await first_module(calls, photo, model, effort)
    if grid is None:
        return best, False
    for step, (rewrite_model, rewrite_effort) in enumerate(lumen.REWRITES, 1):
        content = await refine_with_sides(step, len(lumen.REWRITES), photo, grid, best, calls.witness)
        try:
            text, incomplete = await calls.ask(content, rewrite_effort, rewrite_model)
        except pipeline.ModelFailure as e:
            logger.warning("rewrite {} call failed ({}), keeping the last accepted module", step, e)
            break
        if incomplete:
            logger.info("rewrite {} incomplete ({}), dropped", step, incomplete)
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, calls.witness, rewrite.VIEWS)
        if new_grid is None:
            logger.info("rewrite {} rejected ({}), dropped", step, errors[:1])
            continue
        logger.info("rewrite {} accepted", step)
        best, grid = cand, new_grid
    return best, True


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    """lumen.generate with the sides sheet in every rewrite and same-id resends; its two chains run CONCURRENTLY
    (sedge runs them one after the other), request ids chain c -> k from 40*c, pick/post from 80."""
    witness = CachedWitness(witness)
    photo = image_part(images.photo_data_url(image))
    # Deterministic budget: each concurrent chain's guard sees only its own spend plus a fixed reserve for the rest of
    # the image (never the other chain's progress), so completion order cannot change which calls are made in
    # generation vs replay. The post stage starts from the chains' actual totals, fixed once both have finished.
    chains = []
    for c in range(len(lumen.DRAFTS)):
        ch = RCalls(witness, None, CHAIN_K0 * c)
        ch.credits, ch.calls = float(CHAIN_RESERVE_CREDITS), CHAIN_RESERVE_CALLS
        chains.append(ch)
    finals = list(await asyncio.gather(*(chain(ch, photo, model, effort)
                                         for ch, (model, effort) in zip(chains, lumen.DRAFTS, strict=True))))
    calls = RCalls(witness, None, POST_K0)
    calls.credits = sum(ch.credits - CHAIN_RESERVE_CREDITS for ch in chains)
    calls.calls = sum(ch.calls - CHAIN_RESERVE_CALLS for ch in chains)
    accepted = [m for m, ok in finals if ok]
    if not accepted:
        tried = next((m for m, _ in finals if m), None)
        return await finish(tried, witness, checked=False) if tried else None
    best = accepted[0] if len(accepted) == 1 else await lumen.pick(calls, photo, accepted)
    grid, _ = await check(best, witness, rewrite.VIEWS)
    for model, step, line in POST_STAGES if grid is not None else ():
        content = await refine_with_sides(step, lumen.POST_STEPS, photo, grid, best, witness)
        if line:
            content = [text_part(line), *content]
        try:
            text, incomplete = await calls.ask(content, "high", model)
        except pipeline.ModelFailure as e:
            logger.warning("post refine step {} call failed ({}), keeping the module so far", step, e)
            break
        if incomplete:
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, witness, rewrite.VIEWS)
        if new_grid is None:
            logger.info("post refine step {} rejected ({}), dropped", step, errors[:1])
            continue
        best, grid = cand, new_grid
    logger.info("done · {} calls, {:.0f} credits, {} cached renders", calls.calls, calls.credits, witness.hits)
    return await finish(best, witness)
