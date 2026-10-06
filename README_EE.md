# Early-exit HAR firmware (nRF54L15 DK)

This firmware runs the paper's early-exit pipeline on the nRF54L15 DK with nRF Connect SDK 3.0.2 and TensorFlow Lite
Micro (CMSIS-NN kernels). The pipeline has three parts: a rotation-trained TORUS-9 backbone, three rotation-fitted
exit heads, and per-exit entropy thresholds at lambda = 0.4. As in the previous benchmark (`legacy/`), the firmware
feeds synthetic windows, times preprocessing and inference, and marks both on GPIOs for the PPK2. `EE_INTERFACE.md`
is the contract between the exporter and the firmware: symbol names, arithmetic, and stage/exit numbering.

| backbone | select with | stage tensors | after stage 4 | host check on the real TFLM |
|---|---|---|---|---|
| TCN-Small (default) | nothing, or `EE_BACKBONE=TCN` | int8 | - | self-test 8/8 exits, 8/8 classes |
| DeepConvLSTM-0.25 | `EE_BACKBONE=DCL` | int8 | - | self-test 8/8 exits, 8/8 classes |
| TinierHAR | `EE_BACKBONE=TINIERHAR` | int16 (TFLite 16x8) | float32 C tail: BiGRU + attention + dense (section 2) | self-test 8/8 exits, 8/8 classes |

DeepConvLSTM's last stage is TFLite's fused INT8 LSTM, which keeps its hidden and cell state in variable tensors
that persist from one invoke to the next (your earlier WISDM LSTM in `legacy/` behaves the same way). The trained
model starts every window from zero state, so the firmware resets that stage (`MicroInterpreter::Reset()`, flagged
by `EE_STAGE_STATEFUL` in the generated header) before every invoke; the reset is counted in that stage's time.
Without it, every window reaching the final exit would start from the previous window's LSTM state (host INT8 wF1
0.763 in test order instead of 0.771).

Nothing has run on the board yet. The checks so far were done on the host: the real TFLM (NCS commit 48613f7)
compiled for a 32-bit x86 host, and a Cortex-M33 compile of the sources with your `arm-zephyr-eabi-gcc` 12.2.0. There
has been no `west build` and no link. The board uses the CMSIS-NN kernels. On the host, TFLM was built twice with
NCS 3.0.2's own CMSIS-NN sources: once with CMSIS-NN's generic C code, and once with the DSP code path that the M33
runs, its DSP instructions emulated in C. For all three backbones, int16 TinierHAR included, every stage output
matched the reference kernels byte for byte on 32 windows. CMSIS-NN has not run on silicon (section 2).

## 1. Generate the headers (WSL, repository root)

```bash
source ~/embeddedML/.tf-env/bin/activate
cd ~/personalisation
python export_ee_firmware.py --backbone tcn_small            # or: --backbone tcn
python export_ee_firmware.py --backbone deepconvlstm_0_25    # or: --backbone dcl
python export_ee_firmware.py --backbone tinierhar
python export_ee_firmware.py --help                          # all options
```

The script never trains. It loads the recorded `plain_rot` backbone of one LOSO fold from `results_personalisation/`,
using the repository's `our_method` package. It then fits the three heads on four rotated copies of the training
windows, searches the thresholds, converts four integer stage models (int8; int16 activations for TinierHAR), and
checks that the result reproduces the paper's record. Run it with the GPU visible: `--cpu` works, but the help says it
reproduces the recorded heads only approximately. Each run took 2 to 3 minutes on the GPU. The main options:

| option | default | meaning |
|---|---|---|
| `--subject` | `540` | LOSO fold, i.e. the held-out TORUS-9 user. The backbone was trained on the other users. Records exist for 540, 11113, 26445, 29450, 30609, 45367, 45423 and 91455. |
| `--thresholds` | `rot` | validation windows for the threshold search: rotated (the adopted A4 setting) or `clean` |
| `--lam` | `0.4` | lambda of the threshold search |
| `--n-testvec` | `8` | number of self-test windows written to `ee_<bb>_testvec.h` |

A run writes these files. The headers are never edited by hand; rerunning the exporter overwrites them.

| file | contents | used by |
|---|---|---|
| `src/ee_<bb>_config.h` | window 250 x 7 @ 100 Hz, 9 classes, scaler mean/std, class and channel names, LOSO subject | `main.c` |
| `src/ee_<bb>_model.h` | the four stage `.tflite` arrays and their arena sizes, activation bits (`EE_ACT_BITS`), tap shapes, exit heads (W, b), enabled exits, thresholds, FLOPs per exit, the C tail's weights (`EE_HAS_C_TAIL`, TinierHAR only), op list | `ee_pipeline.cpp` only |
| `src/ee_<bb>_testvec.h` | 8 raw windows of the held-out user, with the exit, class and entropies of the host integer reference | `main.c`, only with `EE_SELF_TEST=1` |
| `ee_export/<bb>/stage0..3.tflite`, `report.json`, `testvec.npz` | the same stage models as files, plus the full report (record check, INT8 wF1, sizes, quantisation) | not built; for reference |

Headers for several backbones can sit side by side in `src/`. Only the selected one is compiled, so only its models
reach flash.

If the stage models exceed the exporter's budgets (`--flash-budget-kb 1024`, `--ram-budget-kb 192`, the latter for the
sum of the arena estimates), the exporter prints `NOT DEPLOYABLE on the nRF54L15 as exported`, writes no headers and exits with code 2.
`--allow-oversize` overrides this. Do not use it for the board.

## 2. TinierHAR: int16 stages and a C tail

TinierHAR differs from the other two backbones in two ways. Both are set in the generated header, so the firmware
needs no TinierHAR-specific option.

**The GRU runs in C (`EE_HAS_C_TAIL 1`).** TinierHAR ends in a bidirectional GRU (16 units per direction), additive
attention pooling over time and a dense softmax. TFLM in NCS 3.0.2 has no GRU kernel, so the GRU can only reach TFLM
as ordinary ops, and both ways of doing that fail:

- Unrolled over its 62 time steps, the GRU alone made stage 3 a 1,052,432 B model (1,071,200 B for all four stages),
  and TFLM needed 214,916 B of arena for it. The application core has 188 KB of RAM (section 9).
- Kept rolled, it converts to a WHILE loop with dynamic-shape SLICE and CONCATENATION, which TFLM aborts on.

So the exporter ends the last stage at the GRU input, `tinierhar_flatten` with shape (1, 62, 56). This section
counts stages from 0, as `EE_INTERFACE.md` and the `ee_init()` log do, so the last stage is stage 3; the breakdown
line calls it `stage4`. The firmware runs the rest in float32 in `ee_math.c` (`EE_INTERFACE.md`, "C tail"):

- `ee_bigru_i16` reads stage 3's int16 output where TFLM left it. It dequantises one time step of 56 values at a
  time, so the 62 x 56 input is never copied to float. Each direction is a Keras GRU (`reset_after`, gates z, r, n)
  starting from zero state. The backward direction runs from the last time step to the first. Both write `hseq`
  [62][32] in time order.
- `ee_attention_pool` scores each time step, takes a softmax over time and returns the weighted sum.
- The dense layer and softmax are the heads' own `ee_linear_head` and `ee_softmax_entropy`. The final exit's class is
  the argmax, and its confidence is that class's probability.

The tail's weights are 7,434 floats (29,736 B of flash). Its static buffers take 8,984 B of RAM: `hseq` (7,936 B),
the per-step scratch, the 62 attention scores and the pooled feature. All sums run in index order. On the 8 self-test
windows, after the real stage models, the C tail's probabilities agree with two NumPy implementations of the same
equations to within 3.6e-7. They also agree with Keras itself to within 3e-7: the exported student's own
`tinierhar_bigru`, `tinierhar_feat` and `softmax` layers, run by TensorFlow on the CPU. The header's tail weights are
bit-identical to the Keras weights. Both GRU directions match Keras' output to within 1.7e-6, so the gate order, the
backward direction and where its outputs land in `hseq` are confirmed against the trained model. Because the tail
recomputes everything from zero state on every window, nothing needs resetting between windows; the host check
confirms that a window gives the same output before and after other windows. Its time is reported separately as
`tail` (section 6). Forcing the final exit (`EE_FORCE_EXIT=3`) runs it. On the x86 host with TFLM's reference
kernels, the tail took 13 to 19 % of a forced-final inference (215 to 280 us of about 1,700 us, over three runs). The
share on the M33 has not been measured. It will differ, because there the stages run on the CMSIS-NN kernels and the
tail on the FPU. Expect it to be larger: the two GRU directions run 124 steps of 3,456 float multiply-adds each, about
0.86 MFLOP or 56 % of the backbone's 1.53 MFLOP (`EE_FULL_FLOPS`), and call `expf` or `tanhf` 5,952 times.

**Activations are int16 (`EE_ACT_BITS 16`, TFLite 16x8).** TinierHAR's first tap is too wide for int8: with int8
activations, exit 1's head agreed with the float head on 84.6 % of held-out windows, against 99.8 % with int16.
Weights stay int8. Stage inputs and outputs, the requantisations, the stats pooling and the tail input are int16, and
the firmware picks the matching tensor type and `ee_math.c` functions at compile time. All stage zero points are 0,
and every stage boundary is a plain copy (6,944 B per copy).

**TRANSPOSE needs a firmware fix in this TFLM.** Stage 3 holds only the Keras Permute + Reshape in front of the GRU:
an int16 TRANSPOSE (perm 0, 1, 3, 2) and a RESHAPE. TFLM 48613f7 (NCS 3.0.2) implements TRANSPOSE for float32 and int8
only. Its Prepare accepts int16, so `AllocateTensors` succeeds, but every invoke then fails with `Type INT16 is
currently not supported by Transpose`. Without a fix, every window that reaches the final exit failed (self-test 6/8,
`ee_run` returning -EIO). There is no CMSIS-NN TRANSPOSE, so the board runs the same kernel. `MicroMutableOpResolver`
offers no way to swap the TRANSPOSE kernel in this TFLM version. `src/ee_tflm_compat.cpp` therefore gives the
interpreters a resolver that forwards to the generated one, but whose TRANSPOSE Eval adds the int16 case using TFLM's
own `reference_ops::Transpose<int16_t>`. That is pure data movement, the same template the stock kernel uses for int8.
Only int16 builds call the wrapper. For TCN and DCL nothing references it. In a host link with Zephyr's section
garbage collection (`-ffunction-sections -fdata-sections`, `--gc-sections`), the TCN and DCL images keep none of the
wrapper, the C tail or the int16 functions. The board's ARM link has not been run. A host probe checks that the
wrapped stage 3 produces exactly the permutation, and that the stock resolver still fails. The cleaner long-term fix
is on the exporter side: fold the permutation into the rows of `EE_GRU_*_W` and leave stage 3 a plain RESHAPE.

The other TinierHAR ops are ADD, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D and RESHAPE. On the host they run in int16
with TFLM's reference kernels and reproduce the exporter's TFLite reference: self-test 8/8 exits and 8/8 classes, max
|dH| 1.2e-7. The board uses CMSIS-NN for the first four. NCS 3.0.2's CMSIS-NN glue has int16 paths for them:
`arm_elementwise_add_s16`, `arm_convolve_wrapper_s16`, `arm_depthwise_conv_s16` and `arm_max_pool_s16`. All
convolution biases in the exported stages are int64, which is what the CMSIS-NN int16 convolution requires. On the
host, the real TFLM with these CMSIS-NN kernels (generic C, and the M33's DSP path with its instructions emulated in C)
reproduced the reference kernels' int16 stage outputs byte for byte on the 8 test vectors and 24 synthetic windows,
and the firmware's self-test passed with them (8/8 exits, 8/8 classes). This has not run on silicon, so the board's
self-test is still the final check.

## 3. Copy to Windows (TFLM must be fetched)

Copy the whole `firmware/` folder, generated headers included, from `\\wsl.localhost\Ubuntu\home\shuhao\personalisation\firmware`
to a local Windows folder. Building straight from the `\\wsl.localhost` share has not been tried. The build compiles
only `src/main.c`, `src/ee_math.c`, `src/ee_pipeline.cpp` and `src/ee_tflm_compat.cpp`; `legacy/` and `ee_export/` are
not built.

TFLM is in the manifest's `optional` group, which is off by default. Your `C:\ncs\v3.0.2` workspace already fetches
it: `.west\config` has `project-filter = +tflite-micro`, and `optional\modules\lib\tflite-micro` is at 48613f7. A
fresh workspace needs `west config manifest.project-filter -- +tflite-micro` followed by `west update`. Without TFLM
the build fails.

## 4. Select the backbone and options

There are three ways to choose a backbone. Use only one of them:

- **west command line**: `-DEE_BACKBONE=TCN|DCL|TINIERHAR` after `--` (see section 5).
- **`src/ee_select.h`**: uncomment one of `EE_BACKBONE_TCN`, `EE_BACKBONE_TINIERHAR` or `EE_BACKBONE_DCL` at the top.
- **`CMakeLists.txt`**: add `target_compile_definitions(app PRIVATE EE_BACKBONE_DCL)`.

If nothing selects a backbone, the build uses TCN. Selecting two is a compile error. Selecting a backbone whose
headers are missing stops with the `#error`.

The options below have their defaults in `src/ee_select.h`. Override them with `-D<NAME>=<value>` after `--`, or edit
the default there.

| option | default | meaning |
|---|---|---|
| `EE_FORCE_EXIT` | `-1` | `-1`: the thresholds decide, which is the deployed policy. `0`, `1`, `2`: always leave at exit 1, 2 or 3. `3`: always run to the final exit. The enabled heads before the forced exit are still evaluated, so their cost is paid (section 8). |
| `EE_SELF_TEST` | `0` | `1`: replay the test vectors before measuring (section 7). This adds about 56 KB of flash. |
| `EE_INFERENCE_IDLE_MS` | `2500` | idle time between measured windows, which gives the PPK2 trace a baseline |
| `EE_WARMUP_ITERATIONS` | `5` | windows run before measuring; they are not logged |
| `EE_USE_WINDOW_MARKER` | `1` | toggle the window GPIO once per window (`0` for a two-pin setup) |
| `EE_LOG_BREAKDOWN` | `1` | log a second line per window with the per-part timings |
| `EE_PART_TIMER_DWT` | `1` | clock for the per-part timings. `1`: the Cortex-M33 DWT cycle counter at 128 MHz, falling back to `k_cycle_get_32()` if it is unusable. `0`: `k_cycle_get_32()`. |

`west build` uses sysbuild by default in NCS 3.0.2. Sysbuild puts a `-D` given after `--` in its own cache, not in
the application's. `CMakeLists.txt` therefore reads every option with `zephyr_get()`, which also covers
`--no-sysbuild` and the image-prefixed form `-D<folder>_EE_BACKBONE=...`. The CMake output confirms what was picked
up:

```
-- Early-exit firmware: EE_BACKBONE=DCL (from the command line)
-- Early-exit firmware: EE_FORCE_EXIT=2 (from the command line)
```

On the board, the `Backbone` and `Force exit` lines of the startup log confirm it again.

## 5. Build and flash (Windows)

Open an nRF Connect SDK 3.0.2 terminal (the nRF Connect for VS Code terminal, or `nrfutil toolchain-manager launch`
as for the previous benchmark) and run:

```bat
cd C:\path\to\firmware
west build -b nrf54l15dk/nrf54l15/cpuapp -p
west flash
```

Other builds:

```bat
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_BACKBONE=DCL
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_BACKBONE=DCL -DEE_FORCE_EXIT=2
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_SELF_TEST=1
```

Always build pristine (`-p`). `-D` values stay in the CMake caches, so a non-pristine rebuild keeps the old ones. Use
the secure `nrf54l15dk/nrf54l15/cpuapp` target. The `/ns` target is not supported: it builds with TF-M, which the RAM
power-down library in `prj.conf` excludes, and it may block the DWT.

At the end of the build, check the memory report (`FLASH` and `RAM` used against region size). Kconfig may print
`... was assigned the value ... but got the value ...` warnings for `prj.conf` lines that do not apply here, for
example `CONFIG_LOG_BUFFER_SIZE`, which exists only in deferred logging mode while this build logs in immediate mode.
These lines come from your previous `prj.conf`, and these warnings do not stop the build. Logs go to RTT: use J-Link RTT Viewer or
the RTT terminal in nRF Connect for VS Code.

## 6. Reading the log

At startup the firmware logs:

- backbone, dataset and LOSO subject;
- window shape, channels and classes;
- the activation precision: `Activations : int8 ...`, or `int16 ... (TFLite 16x8)` for TinierHAR;
- the C tail: `C tail : none ...`, or for TinierHAR
  `C tail : stage 4 output 62 x 56 -> BiGRU(16 per direction) + attention + dense, float32 (7434 parameters, 8984 B buffers)`;
- the policy: lambda, and `rot` = thresholds searched on rotated validation windows;
- one line per early exit: enabled or disabled, entropy threshold, tap shape (T x P), and the full-cost FLOPs of
  leaving there;
- the final exit's FLOPs;
- the force-exit mode, self-test mode, idle time and warm-up count.

`ee_init()` then logs:

- the part timer: `part timer: DWT CYCCNT @ 128000000 Hz`, or a warning followed by the fallback;
- for each stage: model bytes, `arena <used> / <allocated> B used`, and the int8 or int16 input/output element counts
  with the tensors' own scale and zero point;
- whether each stage boundary is a copy or a requantisation;
- for TinierHAR, one `C tail:` line with stage 4's output shape;
- the arena total.

After that comes one line that names both clocks:

```
  Timers        : prep/infer = k_cycle_get_32 @ 1000000 Hz (us); parts = DWT CYCCNT @ 128000000 Hz (0.1 us)
```

Then each measured window logs two lines (`<..>` = microseconds):

```
prep=<P> us | infer=<I> us | exit=final | class=walking (99.6%)
  parts us: quant=<q.q> | stage1=<s.s> stage2=<s.s> stage3=<s.s> stage4=<s.s> | head1=<h.h> head2=<h.h> head3=<h.h> | requant1-2=<r.r> requant2-3=<r.r> requant3-4=<r.r> | other=<o.o> | H1=0.914 H2=0.745 H3=0.322
```

With TinierHAR, a window that reaches the final exit also has `tail=<t.t>` after `stage4`:

```
  parts us: quant=<q.q> | stage1=<s.s> stage2=<s.s> stage3=<s.s> stage4=<s.s> tail=<t.t> | head1=<h.h> ...
```

- **`prep`**: standardisation of one raw window, `(raw - mean) / std` per channel. It is timed while the prep GPIO
  is high.
- **`infer`**: the whole `ee_run()`, timed while the infer GPIO is high. It covers input quantisation, the stages
  that ran, the evaluated heads, the requantisations and the exit decision.
- **`exit`**: `1`, `2` or `3` for an early exit after stage 1, 2 or 3, or `final` after stage 4.
- **`class`**: the predicted class. Its confidence is the softmax probability at an early exit. At the final exit it
  is the dequantised int8 softmax (TCN, DCL) or the C tail's softmax probability (TinierHAR).
- **The `parts` line**, one group per `|`, always in this order. A group is empty when nothing in it ran (after exit
  1: `| head1=.. | | other=..`). The groups are:
  - `quant`: float-to-int8 (int16 for TinierHAR) conversion of the input;
  - `stageN`: each stage invoked;
  - `tail`: TinierHAR's C tail after stage 4 (BiGRU, attention, dense and softmax), only when it ran. `stage4` is then
    only the TRANSPOSE and RESHAPE in front of it;
  - `headK`: each head evaluated (stats pooling, linear layer, softmax and entropy);
  - `requantA-B`: each stage-to-stage requantisation. When both sides share scale and zero point it is a plain
    copy, but it still copies the whole tap (4,000 B for TCN, up to 13,552 B for DeepConvLSTM, 6,944 B for
    TinierHAR);
  - `other`: `infer` minus the sum of the parts (timer reads, mutex, argmax), clamped at 0;
  - `HK`: the entropy (nats) of each evaluated head. Exit K is taken when `HK` is below its threshold.

The two lines use two clocks. `prep` and `infer` come from `k_cycle_get_32()`, which on the nRF54L15 is the 1 MHz
GRTC. They are read inside the GPIO markers, exactly as in the previous benchmark, and are whole microseconds,
floored. The parts come from the part timer:

- **With `EE_PART_TIMER_DWT=1`**, `ee_init()` enables `DWT->CYCCNT` through CMSIS. Zephyr's own DWT support is not
  available because `soc/nordic/nrf54l` does not select `CPU_CORTEX_M_HAS_DWT`. The counter is kept only if it
  exists and counts. It runs at `SystemCoreClock`, which Zephyr sets from the devicetree to 128 MHz, and is printed
  rounded to 0.1 us.
- **With `EE_PART_TIMER_DWT=0`, or after the fallback**, the parts come from the GRTC as well, so their decimal is
  always 0.

`other` mixes the two clocks and is accurate to about +-1 us. Both clocks also count any interrupt that fires inside
a part.

Setting the DWT trace-enable bit powers the core's debug counters. Whether that changes the PPK2 current has not been
measured. Compare the idle baseline of a default build with that of an `EE_PART_TIMER_DWT=0` build. If they differ,
take the energy figures from the `=0` build.

## 7. Self-test (`EE_SELF_TEST=1`)

Before the warm-up, the firmware replays each `EE_TV_RAW` window through the same standardisation and `ee_run`, with
the thresholds deciding (`EE_FORCE_EXIT` is ignored here). It compares the exit, class and evaluated entropies with
the host integer reference:

```
tv  3: label sitting | exit 1 (host 1) | class lying (host lying) | max|dH| 0.00e+00 OK
self-test: exits 8/8, classes 8/8, max |dH| = 1.192e-07
self-test: true-label hits 4/8
```

The `self-test:` summary is the line to check. The real-TFLM host runs gave
`exits 8/8, classes 8/8, max |dH| = 1.192e-07` for TCN, DCL and TinierHAR. Two of TinierHAR's 8 windows reach the
final exit, so they go through the int16 TRANSPOSE fix and the C tail.

- **Small entropy differences** (about 1e-6) can come from `expf`/`logf` on the M33. `ee_math.c` is built with
  `-ffp-contract=off` so that no multiply-adds get fused.
- **An exit or class mismatch** means the headers and the firmware are out of sync, or that the CMSIS-NN kernels on
  the M33 round differently from TFLite's reference kernels. On the host, NCS 3.0.2's CMSIS-NN matched them byte for
  byte, generic C path and emulated DSP path alike (section 2). The exception is a window whose entropy lies very
  close to a threshold; the exporter picks test windows whose evaluated entropies are more than 0.02 nats from the
  thresholds, and whose final-exit class margins are at least 0.05 in probability for the C tail.
- **`ee_run failed (-5)`** on TinierHAR windows that reach the final exit means stage 4's TRANSPOSE failed: the
  int16 wrapper of `ee_tflm_compat.cpp` is not in effect (section 2).
- **`true-label hits`** is plain accuracy on 8 windows, for information only.

Turn the self-test off for footprint and energy measurements.

## 8. Measuring each exit path (PPK2) and the GPIO markers

Synthetic windows are uniform in mean +- 3 std of the training scaler, so they do not look like real activity. Under
the policy they will often run to the final exit. To get the latency and energy of every path, build once for each
value:

```bat
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_FORCE_EXIT=0
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_FORCE_EXIT=1
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_FORCE_EXIT=2
west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_FORCE_EXIT=3
```

The values map to paths as `0` = exit 1, `1` = exit 2, `2` = exit 3 and `3` = final. Add `-DEE_BACKBONE=DCL` for
DeepConvLSTM or `-DEE_BACKBONE=TINIERHAR` for TinierHAR, whose `3` includes the C tail. A forced build still
evaluates the enabled heads before the forced exit, so it has the same cost as reaching that exit under the policy.
Two consequences:

- **A disabled exit can still be forced.** Its head is then evaluated, and a startup warning says so. DeepConvLSTM's
  exit 3 is disabled at lambda = 0.4.
- **The pure backbone without heads** is not one of these builds. Its cost is the sum of the `stageN` and `requant`
  parts, plus `tail` for TinierHAR.

GPIO markers (`app.overlay`, `/zephyr,user`):

| pin | role |
|---|---|
| P1.11 `prep-gpios` | high during preprocessing |
| P1.12 `infer-gpios` | high during inference |
| P1.13 `window-gpios` | toggles once per window |

Each marker goes high before its timer read and low after the second read, so it brackets the timed region.

P1.11 and P1.12 are free on the DK. **P1.13 is push button 0** (`button0`/`sw0`, pull-up, active low, in the NCS 3.0.2
board files). No driver claims it here, but pressing button 0 during a capture pulls the window marker to GND. Move
`window-gpios` in `app.overlay` to a free pin, or set `EE_USE_WINDOW_MARKER=0`, if that matters.

## 9. Memory per backbone

The application core gets 188 KB of RAM and 1428 KB of RRAM. These are `cpuapp_sram` and `cpuapp_rram` in the NCS
3.0.2 devicetree; the remaining 68 KB of RAM and 96 KB of RRAM belong to the FLPR core. The arenas are static RAM, one
per stage. The "used" figures below come from host runs of the real TFLM (32-bit), first with the reference kernels
and then with the CMSIS-NN kernels the board uses. CMSIS-NN requests a little extra scratch. `ee_init()` logs the
board's own figures.

| | TCN-Small | DeepConvLSTM-0.25 | TinierHAR |
|---|---|---|---|
| stage tensors | int8 | int8 | int16 (weights int8) |
| stage models (flash) | 27,848 B | 34,832 B | 22,328 B |
| exit heads W, b (flash) | 5,292 B | 36,396 B | 18,252 B |
| C tail weights (flash) | - | - | 29,736 B (7,434 floats) |
| test vectors (flash, `EE_SELF_TEST=1` only) | 56,192 B | 56,192 B | 56,192 B |
| arenas allocated (RAM) | 79,584 B (19,920 / 19,568 / 19,568 / 20,528) | 135,840 B (52,432 / 35,136 / 26,896 / 21,376) | 121,408 B (40,496 / 30,080 / 30,080 / 20,752) |
| arenas used, host, reference kernels | 53,776 B (13,572 / 13,268 / 13,268 / 13,668) | 104,432 B (42,148 / 27,300 / 20,132 / 14,852) | 110,544 B (37,524 / 29,316 / 29,316 / 14,388) |
| arenas used, host, CMSIS-NN kernels | 54,400 B (13,572 / 13,476 / 13,476 / 13,876) | 106,656 B (42,468 / 27,620 / 20,452 / 16,116) | 110,624 B (37,604 / 29,316 / 29,316 / 14,388) |
| C tail buffers (RAM) | - | - | 8,984 B |
| other static RAM of `ee_pipeline.cpp` (M33 compile) | 1,420 B | 2,608 B | 1,900 B |

The last row covers the four `MicroInterpreter` objects (752 B), the op resolver, the head scratch (`g_feat`,
sized for the widest tap) and the per-stage table. It is the size of `ee_pipeline.cpp`'s static data in the M33
compile, minus the arenas and the tail buffers. The rest of the RAM is configured, not measured:

- two float window buffers in `main.c` (14,000 B);
- the main stack (8,192 B), ISR stack (1,536 B), idle stack (512 B) and system work queue stack (1,536 B);
- the RTT up buffer (4,096 B);
- kernel, driver and libc data.

The configured items add up to 29,872 B. Adding the arenas, the tail buffers and the other static RAM gives the
following, out of 188 KB (192,512 B):

| | TCN-Small | DeepConvLSTM-0.25 | TinierHAR |
|---|---|---|---|
| total before kernel, driver and libc data | 110,876 B | 168,320 B | 162,164 B |
| left for kernel, driver and libc data | about 80 KB | about 24 KB | about 30 KB |

Check the `RAM` line of the memory report, especially for DeepConvLSTM. The TFLM and CMSIS-NN code size is only known
after linking, from the `FLASH` line.

TinierHAR's arenas for the second and third stage (`EE_STAGE1_ARENA_BYTES`, `EE_STAGE2_ARENA_BYTES`; `stage 1` and
`stage 2` in the `ee_init()` log, which counts from 0) leave only 764 B of headroom. The CMSIS-NN kernels request
scratch buffers that the reference kernels do not, but in the host run with NCS 3.0.2's CMSIS-NN (DSP build) those
two stages used exactly as much arena as with the reference kernels: the scratch fits in gaps of the memory plan.
Stage 0 grew by 80 B. The host is 32-bit like the M33, but struct padding can differ by a few bytes. If `ee_init()`
reports `AllocateTensors failed` for a stage, see the next paragraph.

The arena sizes are the exporter's upper-bound estimates. If `ee_init()` logs `AllocateTensors failed; increase
EE_STAGE<s>_ARENA_BYTES`, the stop-gap is to raise that one `#define` in `ee_<bb>_model.h`. That is the only hand
edit that makes sense in a generated header, and the next export overwrites it.

## 10. What is and is not measured

- **Measured:** standardisation (`prep`), and the complete early-exit inference (`infer`) with its per-part
  breakdown.
- **Not measured:**
  - Window acquisition: the synthetic generator runs before the prep marker goes high. It is the same seeded LCG as
    the previous benchmark, but each channel is now uniform in mean +- 3 std of this fold's scaler.
  - Logging, which happens after both markers are low.
  - Pressure detrending. Channel 7 of TORUS-9 is detrended barometric pressure. The detrending is done offline on
    the host before standardisation. A real deployment needs an online detrending filter, and its cost comes on top
    of `prep`.

## Source files

| file | role |
|---|---|
| `CMakeLists.txt` | sources, `-ffp-contract=off` for `ee_math.c`, the west `-D` options (via `zephyr_get()`) |
| `prj.conf`, `app.overlay` | your previous configuration and marker pins. Duplicate `prj.conf` lines were removed. The old file set `CONFIG_MAIN_STACK_SIZE` to 8192 and then to 4096, and the later value won; the main stack is now 8192. |
| `src/ee_select.h` | backbone selection, generated header names, options |
| `src/ee_math.h/.c` | the contract arithmetic in pure C99: standardise, quantise, requantise, stats pooling, linear head, softmax and entropy, argmax (int8 and int16), and the C tail (`ee_bigru_i8` / `ee_bigru_i16`, `ee_attention_pool`) |
| `src/ee_pipeline.h/.cpp` | the TFLM glue: one interpreter and one arena per stage (int8 or int16 tensors, from `EE_ACT_BITS`), the C tail after stage 4 when `EE_HAS_C_TAIL`, `ee_init`, `ee_run`, `ee_get_info`, and the part timer (`ee_part_timer_hz`, `ee_part_timer_name`) |
| `src/ee_tflm_compat.h/.cpp` | int16 TRANSPOSE for TFLM 48613f7 (section 2); used by int16 builds only |
| `src/main.c` | the benchmark: synthetic input, GPIO markers, warm-up, idle, RAM power-down, self-test and logging |
| `src/ee_<bb>_*.h` | generated by `export_ee_firmware.py`; never edited by hand |
| `legacy/` | the previous single-model benchmark, kept byte for byte; not built |
