#!/usr/bin/env python3
"""Draw the README hero, light and dark, and render both to PNG.

    python scripts/build_hero.py            # writes html + png for both themes

The figure states the contribution in one picture: the same utterance is scored
twice, once on a 20 ms grid and once natively on 160 ms blocks, and min-pooling
the fine stream up to 160 ms is worse than predicting 160 ms directly.

Figure discipline, carried over from the sibling repo's heroes:

* One meaning per colour, held across the whole figure set.
      grey   #8FA6BD   inherited machinery: the waveform and the 20 ms stream
      blue   #004488   what this work adds: the native 160 ms head
      red    #B0392B   the thing being located: spoofed audio
* Corner radii stay inside 0.22-0.56 % of canvas width. On 840 px that is
  1.8-4.7 px, so rx=3.
* Solid stroke marks a scored span; nothing here is dashed, because nothing
  here is a crop.
* No gradients, no shadows, no icons. Every mark is a number or a boundary.
"""

from __future__ import annotations

import math
import pathlib
import shutil
import subprocess
import sys

W, H = 840, 300
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "readme"

# The measured numbers this figure exists to show. Sources in README.md.
# These are the ONLY place the figures appear — every label below is derived,
# so the picture cannot disagree with itself.
EER_POOLED = 3.6200          # 20 ms stream min-pooled to 160 ms
EER_NATIVE = 3.1788          # the native 160 ms head
EVAL_SCOPE = "full eval set" # 71,237 utterances, not a slice
DELTA = round(EER_POOLED - EER_NATIVE, 4)

# Cross-corpus, LlamaPartialSpoof: the ordering reverses. The figure says so,
# because a hero that showed only the in-domain win would be overselling a
# result the README then has to walk back.
# SAL's protocol: the crossfade release in full, 76,228 utterances, 20 ms.
XC_POOLED = 27.5033
XC_NATIVE = 30.0332
# Taken from the evaluator, not derived, so the figure cannot disagree with the
# README by a digit. Here the rounded difference happens to agree, but the rule
# stands: this value is copied from the scorer's own output.
XC_DELTA = 2.5299
# The cross-corpus number the README leads with and that SAL's table is
# compared against: the 20 ms fine stream, which is the cell SAL reports.
# NOT XC_NATIVE -- that is the coarse route, which loses off-corpus, and
# putting the losing route's number in the footer would understate the result.
XC_EER_20 = 29.0048

THEMES = {
    "light": dict(bg="#FFFFFF", ink="#16202B", mut="#5A6B7A", faint="#6B7B8A",
                  rule="#E1E7ED", wave="#8FA6BD", blue="#004488", blue_fill="#EAF1FB",
                  red="#B0392B", red_fill="#F7E9E7", box="#8A8A8A"),
    "dark":  dict(bg="#0D1117", ink="#E6EDF3", mut="#9FB0C0", faint="#8B9CAC",
                  rule="#21262D", wave="#5C7590", blue="#6EA8FF", blue_fill="#16243A",
                  red="#E0796B", red_fill="#2A1A18", box="#6E7681"),
}

for _name, _t in THEMES.items():
    for _k, _v in _t.items():
        if not (len(_v) == 7 and _v[0] == "#" and all(ch in "0123456789ABCDEFabcdef" for ch in _v[1:])):
            raise ValueError(f"THEMES[{_name!r}][{_k!r}] is not a hex colour: {_v!r}")

# ---------------------------------------------------------------- geometry
LANE_X0, LANE_X1 = 168, 616          # the grids share one x-range so they align
SCORE_X = 640                         # scores hang off the right of the grids
Y_INPUT, Y_FINE, Y_NATIVE = 112, 168, 216
BAR_H = 26


def envelope(n: int, seed: int = 7) -> list[float]:
    """A deterministic speech-like amplitude envelope in [0.18, 1.0]."""
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        a = 0.55 + 0.42 * math.sin(2 * math.pi * (1.7 * t + 0.11 * seed))
        a *= 0.72 + 0.34 * math.sin(2 * math.pi * (5.3 * t + 0.37 * seed))
        a *= 0.80 + 0.26 * math.sin(2 * math.pi * (11.9 * t + 0.61 * seed))
        out.append(min(1.0, max(0.18, abs(a))))
    return out


def build(theme: str) -> str:
    c = THEMES[theme]
    span = LANE_X1 - LANE_X0

    # The spoofed region: blocks 5..7 of 16, i.e. a contiguous run of 160 ms blocks.
    n_blocks = 16
    bw = span / n_blocks
    spoof_lo, spoof_hi = 5, 8                      # half-open, in block units
    sx0 = LANE_X0 + spoof_lo * bw
    sx1 = LANE_X0 + spoof_hi * bw

    p: list[str] = []
    add = p.append

    add(f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>')

    # ---- title block
    add(f'<text class="t" x="40" y="46">spanmark</text>')
    add(f'<text class="s" x="40" y="70">locating spoofed speech: one model scores every '
        f'20 ms, a second predicts 160 ms blocks natively</text>')
    add(f'<text class="s2" x="40" y="89">in domain the native head wins by {DELTA:.4f} EER '
        f'&#8212; on an unseen corpus pooling wins by {XC_DELTA:.4f}, and that reversal '
        f'is the finding</text>')

    # ---- INPUT: amplitude envelope with the spoofed run marked
    add(f'<text class="l" x="40" y="{Y_INPUT + 14}">INPUT</text>')
    env = envelope(116)
    step = span / len(env)
    for i, a in enumerate(env):
        x = LANE_X0 + i * step
        h = 3 + a * (BAR_H - 3)
        y = Y_INPUT + (BAR_H - h) / 2
        inside = sx0 - 0.01 <= x < sx1
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(step - 0.9, 0.9):.1f}" '
            f'height="{h:.1f}" fill="{c["red"] if inside else c["wave"]}"/>')
    add(f'<rect x="{sx0:.1f}" y="{Y_INPUT - 5:.1f}" width="{sx1 - sx0:.1f}" '
        f'height="{BAR_H + 10}" fill="none" stroke="{c["red"]}" stroke-width="1.4" rx="3"/>')
    add(f'<text class="k" x="{(sx0 + sx1) / 2:.1f}" y="{Y_INPUT - 11:.1f}" '
        f'text-anchor="middle" style="fill:{c["red"]}">spoofed</text>')

    # ---- FINE: the 20 ms grid, 8 segments per 160 ms block
    add(f'<text class="l" x="40" y="{Y_FINE + 13}">20 ms</text>')
    add(f'<text class="k" x="40" y="{Y_FINE + 28}">fine stream</text>')
    n_fine = n_blocks * 8
    fw = span / n_fine
    for i in range(n_fine):
        x = LANE_X0 + i * fw
        inside = spoof_lo * 8 <= i < spoof_hi * 8
        add(f'<rect x="{x:.2f}" y="{Y_FINE}" width="{max(fw - 0.55, 0.5):.2f}" height="18" '
            f'fill="{c["red"] if inside else c["wave"]}"/>')

    # ---- NATIVE: the 160 ms grid, predicted directly
    add(f'<text class="l" x="40" y="{Y_NATIVE + 13}">160 ms</text>')
    add(f'<text class="k" x="40" y="{Y_NATIVE + 28}">native head</text>')
    for i in range(n_blocks):
        x = LANE_X0 + i * bw
        inside = spoof_lo <= i < spoof_hi
        add(f'<rect x="{x + 0.8:.1f}" y="{Y_NATIVE}" width="{bw - 1.6:.1f}" height="18" rx="3" '
            f'fill="{c["red_fill"] if inside else c["blue_fill"]}" '
            f'stroke="{c["red"] if inside else c["blue"]}" stroke-width="1.3"/>')

    # ---- the comparison: pooled vs native, and the gap
    add(f'<text class="n" x="{SCORE_X}" y="{Y_FINE + 14}" style="fill:{c["mut"]}">'
        f'{EER_POOLED:.4f}</text>')
    add(f'<text class="k" x="{SCORE_X + 52}" y="{Y_FINE + 14}">pooled</text>')
    add(f'<text class="n" x="{SCORE_X}" y="{Y_NATIVE + 14}" style="fill:{c["blue"]}">'
        f'{EER_NATIVE:.4f}</text>')
    add(f'<text class="k" x="{SCORE_X + 52}" y="{Y_NATIVE + 14}">native</text>')
    # the bracket joining the two scores, annotated with the delta
    bx = SCORE_X + 104
    add(f'<path d="M{bx} {Y_FINE + 9} H{bx + 7} V{Y_NATIVE + 9} H{bx}" fill="none" '
        f'stroke="{c["blue"]}" stroke-width="1.3"/>')
    # anchored to the right margin so the label can never run off the canvas
    add(f'<text class="c" x="800" y="{(Y_FINE + Y_NATIVE) / 2 + 13:.0f}" text-anchor="end" '
        f'style="fill:{c["blue"]}">&#8722;{DELTA:.4f}</text>')

    # ---- footer
    add(f'<line x1="40" y1="256" x2="800" y2="256" stroke="{c["rule"]}" stroke-width="1"/>')
    add(f'<text class="n" x="40" y="281">segment EER {EER_NATIVE:.4f} at 160 ms '
        f'on PartialSpoof</text>')
    add(f'<text class="k" x="404" y="281">{EVAL_SCOPE} &#183; cross-corpus '
        f'{XC_EER_20:.2f} at 20 ms &#183; two routes, SHA-gated &#183; MIT</text>')

    body = "\n  ".join(p)
    return f"""<meta charset="utf-8">
<style>
 @page {{ size: {W}px {H}px; margin:0; }}
 html,body {{ margin:0; padding:0; width:{W}px; height:{H}px; background:{c["bg"]}; }}
 svg {{ display:block; }}
 .t  {{ font:700 22px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["ink"]}; }}
 .s  {{ font:400 14px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["mut"]}; }}
 .s2 {{ font:400 12px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["faint"]}; }}
 .l  {{ font:700 11px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["mut"]};
        letter-spacing:.08em; }}
 .c  {{ font:700 15px "Liberation Sans","DejaVu Sans",Arial,sans-serif; }}
 .k  {{ font:400 11px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["faint"]}; }}
 .n  {{ font:700 13px "Liberation Sans","DejaVu Sans",Arial,sans-serif; fill:{c["ink"]}; }}
</style>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  {body}
</svg>
"""


def render(html_path: pathlib.Path, png_path: pathlib.Path) -> bool:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        print("  no chrome/chromium on PATH; wrote HTML only")
        return False
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
         f"--screenshot={png_path}", f"--window-size={W},{H}",
         "--default-background-color=00000000", html_path.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    return png_path.exists()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        html = OUT / f"hero-{theme}.html"
        png = OUT / f"hero-{theme}.png"
        html.write_text(build(theme))
        ok = render(html, png)
        size = f"{png.stat().st_size / 1024:.1f} KB" if ok else "-"
        print(f"  hero-{theme}: {html.name} -> {png.name} {size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
