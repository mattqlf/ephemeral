# Ephemeral — Speaker Script

> Each section maps to a slide. ~10 minutes total.
> Strategy: slides show the technical terms; the speaker explains them in plain language.

---

## Slide 1 — Title (Hero)

> *Let the image and title sit for a beat.*

---

## Slide 2 — Google just shipped Project Genie

Last week, Google DeepMind publicly launched Project Genie — a product built on their Genie 3 world model. You give it a text prompt or image, and it generates an interactive 3D-like world you can navigate in real time at 20 to 24 FPS, 720p.

It's impressive. But there are real limitations. Sessions are hard-capped at 60 seconds — a compute cost constraint. It's closed source, closed weights. You need a Google AI Ultra subscription at $250 a month. U.S. only.

Google proved that real-time interactive world generation works — then locked it behind a paywall.

---

## Slide 3 — We made it open source

We built the same thing and made it open source. Open weights, open code, no subscription, no time limit. You run it on your own hardware. Resolution is a bit lower — 352 by 640 — but we hit 25 FPS. And you can modify it, extend it, train your own variants.

---

## Slide 4 — Three-stage pipeline

The system is built in three stages.

Stage 1 — we start with Wan 2.1. It's a video generation model with 14 billion parameters — similar in scale to the models behind ChatGPT. DiT stands for Diffusion Transformer — it's the architecture that generates video. We strip out its text understanding and retrain it purely on visual dynamics — how the world looks and moves.

Stage 2 — we add controls. Action modules are small neural networks that take your keyboard and mouse input and feed them into the model. We train these on about 800 hours of gameplay footage that's been annotated with what keys were pressed at each moment.

Stage 3 — we make it real-time. The original model takes 30 steps to generate a batch of frames — way too slow. Self-Forcing is a training technique that compresses that down to just 3 steps, fast enough to stream.

---

## Slide 5 — Wan 2.1 backbone

The 3D Causal VAE is basically a video compressor. It shrinks the video down — 8 times smaller spatially, 4 times temporally — into what we call "latent space." Think of it as a compact representation the model can work with efficiently, rather than processing every raw pixel.

The DiT — the Diffusion Transformer — is a 30-layer neural network. Same type of architecture as ChatGPT, but instead of predicting the next word, it's trained to generate video. Flow matching is the training method — it learns how to gradually turn random noise into clean video frames.

We removed text conditioning entirely. That means the model can't rely on language to understand what's happening — it has to learn physics and movement purely from watching video.

---

## Slide 6 — The action module

This is how the user controls the world.

Keyboard input is discrete — you're either pressing W or you're not. Multi-hot just means multiple keys can be pressed at once. These get injected into the model through cross-attention, which is how the model pays attention to the action signal alongside the video.

Mouse input is continuous — smooth deltas that represent how much you moved. These go through an MLP, which is a small neural network, and temporal self-attention, which helps the model understand how the camera movement evolves over time.

The key insight: we only install these modules in the first 15 of 30 transformer blocks. Early layers decide where things are and how they move — that's action-dependent. Later layers handle textures and fine details — not action-dependent. So we skip them. Halves the compute cost with barely any quality difference.

---

## Slide 7 — From bidirectional to streaming

Here's the core challenge. Bidirectional attention means every frame can see every other frame — including ones that haven't happened yet. That's great for video quality, but it means you'd need to generate the entire video before you can show a single frame. Not compatible with real-time streaming.

Self-Forcing is the solution. Instead of training the model on perfect data, we have it practice generating from its own imperfect outputs. It learns to stay stable over time, even when small errors accumulate. Think of it like learning to ride a bike by actually riding — not by watching someone else ride perfectly.

The result: 30 denoising steps compressed to 3, and the model can now stream using a KV-cache.

---

## Slide 8 — KV-cache & rolling memory

KV-cache stands for key-value cache — it's the model's short-term memory. It stores information about recent frames so the model doesn't need to reprocess everything from scratch each time.

Block-wise causal means: within each group of 3 frames, the model can look freely in all directions. But across groups, it can only look backwards — never into the future. This is what makes streaming possible.

The cache holds about one second of video — 6 frames. As new frames come in, old ones are discarded. Memory stays constant no matter how long you play.

---

## Slide 9 — From image to interactive world

This is where it all comes together.

You provide a single image. CLIP — an image understanding model — analyzes it and extracts what's in the scene. The 3D VAE compresses it into the starting state. That's all the model needs to begin.

From there it's a loop. Your keyboard and mouse inputs arrive via WebSocket. They get embedded — converted into numbers the model understands — and injected into the first 15 transformer blocks. The model runs 3 denoising steps and produces 3 new frames. The VAE decompresses them back into actual pixels. They get sent to your browser. 120 milliseconds per batch, 25 frames per second. Then it waits for your next input and does it again. Indefinitely.

---

*End of technical section.*
