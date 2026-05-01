# Auto-EQ LSTM: Mode Collapse Diagnosis

## Observed behavior

The auto-EQ LSTM controller outputs a near-constant correction regardless of input
material. Diagnostic readout from the plugin GUI showed the 5-band gains locking to
approximately the same values within the first few blocks and holding there for all
material tested (full mix, bass, drums, vocals):

```
LS  @ 1010 Hz : -1.7 dB
110 Hz        : -0.1 dB
1.1 kHz       : -2.6 dB
7   kHz       : -1.7 dB
HS  @ 10 kHz  : -1.5 dB
```

Phase cancellation testing (pre-EQ track vs. post-EQ track inverted) also confirmed
the model was not adapting — the cancellation was clean when EQ mix was at 0%, and
slightly imperfect when the model was active, but the residual was constant across
material.

## Root cause: collapse to the conditional mean

The LSTM learned to output the **average optimal correction** across the training set
rather than a material-dependent correction. This is the classic MSE mode-collapse
failure mode: given a distribution of (input, target) pairs, minimizing mean squared
error in the parameter space encourages the model to predict the conditional mean of
the targets. If the training targets have high variance across material types but low
variance in their *average*, the network converges to outputting that average for
everything.

Concretely: if the training pipeline pairs `(raw_audio_block, ideal_eq_params)` and
the loss is MSE over the 15 sigmoid parameter channels, the LSTM minimizes loss by
learning the mean correction (a broadband -1 to -3 dB cut) and ignoring material
variation.

## Contributing factors

### 1. Per-block peak normalization

At runtime we normalize each 128-sample block to 0.5 peak before feeding the LSTM:

```cpp
const float ctrl_scale = (peak > 1e-6f) ? (0.5f / peak) : 1.f;
```

This was added to match training-time normalization in `prepare_auto_eq_data.py`
(line 258-260: `dry = (seg * (0.5 / seg_peak)).astype(np.float32)`). However the
training normalization was applied **per segment** (likely longer than 128 samples),
not per block. Normalizing every 128-sample block independently discards the
amplitude envelope and reduces spectral variation between blocks, making all inputs
look more similar to the model and further collapsing its discriminative ability.

### 2. Short temporal context

At 128-sample blocks (~3 ms at 44100 Hz), the LSTM only sees a very short window of
audio per step. Discriminating material classes (e.g. bass vs. full mix) reliably
requires building up a spectral profile over hundreds of milliseconds. If the model's
hidden state saturates to a fixed attractor early and the loss doesn't penalize this,
the LSTM stops updating its state meaningfully after the first few blocks.

### 3. Loss function doesn't penalize homogeneous output

An MSE loss on predicted EQ parameters against targets does not explicitly punish the
model for outputting the same correction to different material. Adding a
**discrimination loss** or training with **contrastive pairs** (same material →
similar correction, different material → different correction) would directly
incentivize adaptation.

## Recommended training fixes

### Fix 1: Verify normalization alignment

Check whether `prepare_auto_eq_data.py` normalizes per segment or per block. If per
segment, change the runtime to normalize over a longer window (e.g. an exponential
moving average of peak over ~500 ms) rather than per 128-sample block. This preserves
the amplitude envelope information the model was trained to see.

### Fix 2: Longer conditioning context

Feed the LSTM multiple concatenated blocks or add a longer strided input so it sees
~500 ms of audio before producing a parameter estimate. Alternatively, train with
TBPTT over longer sequences so the hidden state has more time to differentiate
material before the gradient is computed.

### Fix 3: Add a discrimination or diversity loss

Add a regularization term that penalizes outputting similar parameters for
dissimilar-class inputs. The simplest form: during training, sample pairs of examples
from different material classes and add a loss term that pushes their predicted EQ
parameter vectors apart. A contrastive or triplet loss on the parameter embedding
works well here.

### Fix 4: Per-class fine-tuning with held-out material

Rather than training a single model across all classes, train the per-class models
(bass, drums, vocals, full_mix) exclusively on their own class with a much smaller
learning rate for the final few epochs. This forces each class controller to specialize
rather than converge to the global mean.

### Fix 5: Target diversity check

Before retraining, compute the variance of `ideal_eq_params` across material classes
in the training set. If the variance is low (all classes have similar targets), the
dataset itself may be the problem — the source material may be too homogeneous or the
reference processor may not be applying meaningfully different EQ to different classes.
