# ROCm backend notes for this candle fork

Branch `rocm-backend-fixes`. Published so the AMD Radeon / ROCm work in
[`Prog-Newb/SceneWorks`](https://github.com/Prog-Newb/SceneWorks) and
[`Prog-Newb/Inference-Rocm`](https://github.com/Prog-Newb/Inference-Rocm) has a
resolvable candle dependency.

**Most of this repository is not my work.** Read the attribution section before
assuming otherwise.

## Attribution

Three layers, oldest first:

**1. candle itself** — [`huggingface/candle`](https://github.com/huggingface/candle),
dual MIT / Apache-2.0. Several hundred contributors; the bulk by Laurent Mazare
(~1,600 commits across two identities) and Nicolas Patry (~279). `LICENSE-MIT`
and `LICENSE-APACHE` in this tree are unmodified.

**2. The ROCm backend** — [`n1gh7e0us/candle`](https://github.com/n1gh7e0us/candle),
also Apache-2.0. Upstream candle has **no ROCm backend at all**. Everything under
`candle-core/src/rocm_backend/` and the whole `candle-rocm-kernels/` crate
(which carries the `.hip` kernels) originates there. This repo is a fork of it,
not of candle directly, so GitHub's fork lineage reflects that. **The person who
made ROCm work on candle is n1gh7e0us, not me.**

**3. My 9 commits**, listed below. They are fixes and performance work on top of
that backend — not the backend.

### On my two git identities

My commits appear under two author names, both mine:

- `Chris <chris@localhost>` — 7 commits
- `Franklin Mint <local@localhost>` — 2 commits

Same person; I committed from two machines with different git configs and have
left them unsquashed rather than rewrite pushed history. Both use `@localhost`
addresses, which nothing else in this repository does, so the two identities are
unambiguous.

**Do not confuse these with similarly-named upstream candle contributors.** The
following are *other people*, unrelated to me, and their commits are their own:

| Author | Email |
|---|---|
| Chris Ha | `hac541309@gmail.com` |
| Christian Loehnert | `krampenschiesser@users.noreply.github.com` |
| Christian Balcom | `robot.inventor@gmail.com` |
| Christopher Fleetwood | `45471420+FL33TW00D@users.noreply.github.com` |

A `git log --author=Chris` will match all of them. Filter on the email, not the
name.

## My commits

Oldest first.

| Commit | Author identity | What it does |
|---|---|---|
| `a0e066bc` | Franklin Mint | Working ROCm gfx1100 GPU kernel execution. |
| `7c5ce6b5` | Franklin Mint | Validate ROCm elementwise reduction and matmul. |
| `b264aac7` | Chris | Fix element-vs-byte `start_offset` handling; make conv2d stride-safe. |
| `8d6c4444` | Chris | `ROCM_CONV_DEBUG` env probe logging conv2d/conv1d problem dims before the MIOpen call, to locate a naive f64 conv hang. |
| `11459c03` | Chris | Portable ROCm SDK path for the sidecar; bf16 binary/unary ops. |
| `b85d5c51` | Chris | Fill the bf16 kernel chain needed for SeedVR2 image and video upscale. |
| `14d3cc03` | Chris | Cache the chosen MIOpen conv algorithm instead of re-searching every call. |
| `8fec013b` | Chris | **Reverts `14d3cc03`.** See below. |
| `c2d37f62` | Chris | Merge candle `1e6aa85e`, the revision SceneWorks 0.9.0 vendors. |

### The MIOpen dead end, recorded on purpose

`14d3cc03` and `8fec013b` are kept as a pair rather than squashed away, because
the failed attempt is the useful part.

`miopenFindConvolutionForwardAlgorithm` does not merely *return* an algorithm —
it registers an invoker as a side effect. Caching the returned algorithm id and
skipping the search on later calls looks like an obvious win and is not: the
invoker cache is bounded and evicting, so a cached id eventually refers to an
invoker that no longer exists. Hence the revert.

The cost this was trying to avoid is real. A cold-shape exhaustive search was
measured at **321.82 s** on a first tile. The find-db at
`%USERPROFILE%\.miopen\gfx1100_48.HIP.*.ufdb.txt` is what makes later runs fast,
which also means **any timing taken against a warm find-db is not a cold-start
number.** Two separate measurements in this project were invalidated by exactly
that confusion.

## Scope of validation

gfx1100 (RX 7900 XTX) only, ROCm 7.2.1, Windows. No other architecture has been
run. The HIP kernels compile at **runtime**, not build time —
`candle-rocm-kernels` has no build script; `src/compile.rs` compiles on first use
and caches per architecture under
`%LOCALAPPDATA%\candle-rocm\{arch}-{rocm_version}`. Architecture resolves from
`CANDLE_ROCM_ARCH`, then `option_env!`, then `rocminfo`. So a built binary is not
architecture-locked, but a first run on a cold cache pays compilation.

## Relationship to the other two repos

| Repo | What it holds |
|---|---|
| this one | candle + n1gh7e0us's ROCm backend + my fixes |
| [`Prog-Newb/Inference-Rocm`](https://github.com/Prog-Newb/Inference-Rocm) | `runtime-rocm` bundle and the SeedVR2 engine work (tiler, conv3d, GroupNorm, dtype fix) |
| [`Prog-Newb/SceneWorks`](https://github.com/Prog-Newb/SceneWorks) branch `amd-rocm-0.8.7` | SceneWorks-side plumbing: backend selection, worker shim, capabilities, job routing |

All three are needed for the ROCm lane; none is sufficient alone.

## License

Unchanged from upstream: dual MIT / Apache-2.0. `LICENSE-MIT` and
`LICENSE-APACHE` are as received.

Per Apache-2.0 section 4(b), the statement of changes is the commit table above.
All other commits in this history belong to their original authors in
`huggingface/candle` and `n1gh7e0us/candle`.
