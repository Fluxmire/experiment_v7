# Three.js reconstruction miner: merged v5 (research loop 7)

Image-to-Three.js generation pipeline for the 404-GEN GPT competition. Root `main.py` exposes
`async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None` (the same contract as the
base `Olauge-Secret/sedge-wren@fc79ce3`). The pipeline is in `miner/merge.py`; everything else is the base miner,
unchanged (see `provenance.json` for per-file sha256).

## Pipeline (about 10 model calls per image)
1. **Two draft chains, run concurrently** (from sedge-wren):
   - Astra-medium and Sol-medium drafts, each render-checked with up to 2 repairs;
   - each chain then gets a Sol-high rewrite and an Astra-high rewrite, each kept only if it renders.
2. **Pick:** an Astra-low penalty panel on the judge's 4 front views.
3. **Post refines:** Sol high → Astra high → **Sol high (step 3 of 3)**, each kept only if it renders.
4. **Finish:** wrap + flat-object tilt, rendered once (plain-wrap fallback).
5. **Merged from `SmoothTight/newvictor@0f4bc93`** (re-implemented): every rewrite and refine turn also shows the
   module's SIDES sheet (right / back / left / top, 256 px). The instruction: complete any back or side that is
   missing, flat or hollow.

**Engineering:**
- Structural request ids: chain 1 from 1, chain 2 from 4001, pick/post from 8001.
- A deterministic per-chain budget guard, so completion order never changes which calls are made.
- Same-id resend on dropped connections (≤3×).
- A per-image render cache (the renderer is deterministic).

## Budget
- **Per image:** about 10 calls (max 11), about 2.75M credits (max about 4.05M) in local tests; stays under
  5M / 60 calls.
- **24 h fit:** about 2,100 s of model time per image, so 128 images ≈ 12.5 h at 6 concurrent calls.

## Local evaluation (round 2 sample, 32 stems, local Stage-1 judge only)
Pooled over research loops 7 and 8 (two independent runs of this exact pipeline), 64 duels per opponent:

| Opponent | Net |
|---|---|
| leader `huypg-zenai/pgh3` | **+20 (+31%)** |
| `Olauge-Secret/sedge-wren` | −6 |
| `SmoothTight/newvictor` | +6 |
| `taohunter251/amber-wren` | +9 |

The local judge is a proxy for the official 4-stage judge.
