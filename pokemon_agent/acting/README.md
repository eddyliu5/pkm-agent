# Acting Package

Purpose: convert evaluated candidates into one legal immediate simulator selection.

Target behavior: acting chooses from scored candidates, samples only legal candidate first selections in train mode, sanitizes the selected indices, and applies fallback guardrails only after evaluation. Every override should carry an explicit reason.

Trainable modules: none. Metrics are invalid-action rate, fallback override rate, raw-scorer choice rate with guardrails disabled, and train-mode legal-sampling rate.
