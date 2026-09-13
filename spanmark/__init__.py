"""spanmark — segment-level localization of partially spoofed speech.

Two routes, one deployment. A fine model scores every 20 ms; a second model
predicts 160 ms blocks natively. `checkpoints/active.txt` names the native
model, whose `deployment` block names the fine partner and verifies its SHA-256
before inference — see `spanmark.runtime.checkpoint`.
"""

__version__ = "0.1.0"
