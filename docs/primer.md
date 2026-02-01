# Audio Restoration 101: A Primer for Conference Recordings

If you have ever strained to understand a speaker in a noisy recording, you already know why this field exists. Conference audio is where good ideas go to die. The content might be brilliant, but the recording is distant, reverberant, noisy, or all three. The goal of restoration is not to make it sound like a studio album. The goal is simpler and harder: make it **easy to understand** and **pleasant to listen to** for long stretches of time.

This primer starts at first principles and builds toward a practical, modern pipeline. We will use technical terms, but we will translate them immediately. You do not need to be an audio engineer to follow along.

---

## 1) Sound Is Air Pressure, Not Magic

Sound is just air pressure changing over time. A microphone turns those pressure changes into a voltage. An analog‑to‑digital converter samples that voltage and produces numbers. Those numbers are a **waveform**: a list of sample values, taken many times per second.

Two ideas matter right away:

- **Sample rate**: how often you take a measurement. 16,000 samples per second (16 kHz) captures frequencies up to about 8 kHz; 44,100 samples per second (44.1 kHz) captures up to ~22 kHz. Speech is mostly below 8 kHz, but consonant clarity often lives between 8–12 kHz.
- **Amplitude**: how big the waveform is. Bigger amplitude sounds louder. If it is too big, it clips and distorts.

Every restoration algorithm is just a different way of changing that list of numbers.

**Example**: imagine a pure 440 Hz tone (a simple sine wave). If you sample it at 8 kHz, you can represent it pretty well. If you sample it at 1 kHz, it gets mangled. Sample rate sets the ceiling for what you can represent.

---

## 2) Why Conference Audio Is So Hard

Most conference recordings are made in rooms that were designed for people, not microphones. That creates a predictable set of problems:

- **Noise**: HVAC hum, laptop fans, crowd rustle, projector hiss, traffic outside.
- **Reverberation**: the room bounces sound. Consonants blur. Speech becomes “muddy.”
- **Distance**: far microphones capture more room than voice, so you hear the space more than the speaker.
- **Inconsistent levels**: people speak at different volumes, move away from the mic, or turn their head.

The combination is brutal. You end up with a recording where every sentence is technically present but psychologically exhausting to follow.

---

## 3) What “Good” Actually Means Here

For conference audio, “good” does not mean flat frequency response or audiophile purity. It means:

- You can understand every sentence without effort.
- You can listen for an hour without fatigue.
- The background is quieter, but the voice still sounds human.

This is why restoration is a speech‑first problem. If you optimize for music‑style fidelity, you can easily make speech worse.

**Concrete test**: can you transcribe the recording accurately without pausing and rewinding? If yes, you are close to “good.”

---

## 4) The Basic DSP Toolkit (and Why It Still Matters)

Before neural models, people used classic signal processing. It is still the foundation.

### High‑pass and low‑pass filters

- **High‑pass** removes very low rumble (below ~80–120 Hz). This cleans HVAC and table thumps.
- **Low‑pass** removes very high hiss (above ~10–12 kHz). This reduces noisy air and projector squeal.

These are blunt tools, but they are fast and reliable.

### Compression

Compression narrows the dynamic range: quiet parts get louder, loud parts get tamed. This makes speech more consistent and less fatiguing.

### Loudness normalization

Normalization sets the overall level to a target (often ‑16 LUFS for speech). This ensures consistent volume across recordings.

**Example**: A raw recording might swing between whispers and shouts. Compression and normalization smooth those swings so the listener does not ride the volume knob.

---

## 5) Why DSP Alone Is Not Enough

Classic filters cannot distinguish “noise” from “speech.” If you low‑pass too hard, you erase consonants. If you compress too aggressively, you bring up the noise floor.

This is why modern systems move into the **time‑frequency domain** using a **spectrogram**. Instead of viewing audio as a single waveform, you view it as “how much energy exists at each frequency, at each moment.”

That view makes it possible to say: “This cluster is probably noise, remove it. This cluster is speech, keep it.”

---

## 6) Spectrograms and the Core Trick of Noise Reduction

A spectrogram is created by slicing the audio into small windows (e.g., 20–50 ms), running an FFT on each slice, and stacking the results. You end up with a time‑frequency image.

Noise reduction often works like this:

1. Find a segment that contains mostly noise (no speech).
2. Estimate its frequency profile (what the noise “looks” like).
3. Subtract that profile from the rest of the audio.

This can be done with classical spectral gating or with neural models that have learned more complex patterns.

**Example**: if the noise has a constant 60 Hz hum and a steady hiss, the algorithm can suppress those frequencies across time while preserving speech harmonics.

---

## 7) Why Voice Activity Detection (VAD) Is Crucial

If your noise estimate is contaminated by speech, you will subtract speech. That is how you get robotic, hollow audio.

VAD solves this by labeling frames as speech or non‑speech. Then you estimate noise only from non‑speech regions. This makes denoising far more accurate.

There are two common approaches:

- **Energy‑based VAD**: simple and fast, but confused by noisy rooms.
- **Neural VAD** (e.g., Silero): more accurate, especially in difficult recordings.

If you want one “big win” that is not heavy, VAD‑guided noise estimation is it.

---

## 8) Why Neural Denoisers Are a Big Deal (and a Risk)

Neural denoisers learn what speech looks like from massive datasets. Instead of “subtracting noise,” they can reconstruct speech and suppress what does not fit the pattern.

This is why models like DeepFilterNet often sound significantly better than classical gating.

But there is a trade‑off: if the model is too aggressive, it can create artifacts (warbling, metallic tones, missing consonants). You win on noise but lose on speech clarity.

**The lesson**: neural denoising is powerful but should be followed by a quality gate and careful tuning.

---

## 9) Dereverb Is a Different Problem

Noise reduction removes unwanted sounds. **Reverberation is not noise**. It is delayed versions of the speech itself. That makes it harder to remove without harming the voice.

This is why dereverb is a separate stage. Techniques like WPE (Weighted Prediction Error) attempt to model late reflections and subtract them. Neural dereverb models can be even stronger, but often require more compute.

If you remove noise but leave reverb, you get “clean but muddy.” If you remove reverb but leave noise, you get “clear but gritty.” You need both.

---

## 10) A Practical Modern Pipeline

A strong, real‑world pipeline for conference audio looks like this:

1. **Load, resample, and convert to mono** (consistency matters).
2. **Run VAD** to identify non‑speech segments.
3. **Estimate noise from non‑speech** (better subtraction).
4. **Neural denoising** for high‑quality noise suppression.
5. **Dereverb** to reduce room smear.
6. **Light EQ + compression + loudness normalization** for polish.
7. **Quality checks** to ensure no regression.

This is not over‑engineering. This is the minimal structure needed to get reliable, high‑quality results across messy recordings.

---

## 11) Metrics: How You Keep Yourself Honest

You need metrics. Otherwise you will convince yourself that “it sounds better” because you want it to.

Useful metrics:

- **SNR**: did the noise floor actually drop?
- **DNSMOS**: a learned proxy for perceived quality (useful for speech).
- **STOI / PESQ**: intelligibility and perceptual quality when a clean reference exists.
- **Listening tests**: the final authority.

A **quality gate** (fail if quality regresses) is one of the most important pieces of engineering discipline in this field.

---

## 12) Constraints Define “Best Possible”

“Best possible” always means “best under real constraints.” Typical constraints include:

- **Compute**: GPU or CPU only?
- **Latency**: real‑time, near‑real‑time, or offline batch?
- **Dependencies**: can you ship heavy models or not?

A GPU lets you run higher‑quality neural models. CPU‑only pushes you toward efficient algorithms. Real‑time processing limits the complexity of your stages.

The best project is not the fanciest. It is the one that **delivers the best intelligibility under the constraints you actually have**.

---

## 13) A Working Definition of Success

If you want a sentence to guide you:

**A great conference‑audio restoration tool makes speech easy to understand for long stretches of time, without introducing artifacts, and does so reliably on real recordings.**

Everything else is implementation detail.

---

## 14) The Smallest Next Step That Actually Helps

If you already have a pipeline and want the highest impact per unit effort, do this:

- Add **VAD‑based noise estimation**.
- Add **adaptive noise‑reduction strength** (tuned per recording).
- Keep the rest of your chain stable.

This tends to deliver a real, audible jump without risky changes.

---

## 15) Where to Go Deeper

If you want to keep going, focus on these:

- How spectral gating behaves in different SNR conditions.
- Dereverb methods (WPE and neural dereverb) and when they fail.
- Speech intelligibility metrics and their limitations.

The intuition you build here will matter more than any single model.
