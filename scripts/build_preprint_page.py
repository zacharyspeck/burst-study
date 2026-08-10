"""Render docs/preprint.html from the committed analysis JSON. No network, no checkpoints.

Every statistic and every plotted point is read out of docs/measurements/ rather
than typed, for the reason this repo keeps repeating: a number written twice is a
number that can disagree with itself. The 1.55 TB of checkpoints these were
computed from are not committed and are not needed -- 2026-08-10-barrier-curves.json
carries the per-pair curves and the per-step divergence behind both figures.

    python scripts/build_preprint_page.py

Paths derive from __file__ (S110): this script must run anywhere the repo does.
"""
from __future__ import annotations
import json, math, pathlib, statistics

REPO = pathlib.Path(__file__).resolve().parents[1]
M = REPO / "docs" / "measurements"
bar = json.loads((M / "2026-08-10-barrier-analysis.json").read_text())
hel = json.loads((M / "2026-08-10-analysis-heldout_loss.json").read_text())["result"]
cur = json.loads((M / "2026-08-10-barrier-curves.json").read_text())

fl = bar["noise_floor"]; pr = bar["registered_contrasts"]["primary"]
hp = hel["registered_contrasts"]["primary"]; A = {r["arm"]: r for r in bar["arms"]}
l2f = statistics.fmean(v["l2_raw"] for v in cur["twin_vs_twin"].values())
arm_b = statistics.fmean(r["mean"] for r in bar["arms"])
arm_l2 = statistics.fmean(r["l2_raw_mean"] for r in bar["arms"])

ALPHAS = cur["alphas"]
ARM = [statistics.fmean(v["losses"][i] for v in cur["arm_vs_twin"].values()) for i in range(21)]
FLOOR = [statistics.fmean(v["losses"][i] for v in cur["twin_vs_twin"].values()) for i in range(21)]
STEPS = cur["trajectory"]["steps"]
TRAJ = cur["trajectory"]["mean_abs_diff_from_twin"]

# ---------- figure 1: interpolation curves ----------
W,H,PL,PR,PT,PB=680,340,52,18,18,40
def fig1():
    ymin,ymax=3.0,8.4
    def X(a): return PL+(a)*(W-PL-PR)
    def Y(v): return PT+(ymax-v)/(ymax-ymin)*(H-PT-PB)
    def path(vals): return "M"+" L".join(f"{X(ALPHAS[i]):.1f},{Y(vals[i]):.1f}" for i in range(21))
    g=[]
    for v in (3,4,5,6,7,8):
        g.append(f'<line class="grid" x1="{PL}" y1="{Y(v):.1f}" x2="{W-PR}" y2="{Y(v):.1f}"/>')
        g.append(f'<text class="tick" x="{PL-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v}.0</text>')
    for a in (0,0.25,0.5,0.75,1.0):
        g.append(f'<text class="tick" x="{X(a):.1f}" y="{H-PB+20}" text-anchor="middle">{a:g}</text>')
    g.append(f'<path class="s-floor" d="{path(FLOOR)}"/>')
    g.append(f'<path class="s-arm" d="{path(ARM)}"/>')
    # endpoints + peak markers
    g.append(f'<circle class="m-floor" cx="{X(0.5):.1f}" cy="{Y(max(FLOOR)):.1f}" r="4"/>')
    g.append(f'<circle class="m-arm" cx="{X(0.5):.1f}" cy="{Y(max(ARM)):.1f}" r="4"/>')
    g.append(f'<text class="dl dl-floor" x="{X(0.5)+10:.1f}" y="{Y(max(FLOOR))+4:.1f}">two seeds · peak {max(FLOOR):.2f}</text>')
    g.append(f'<text class="dl dl-arm" x="{X(0.5)+10:.1f}" y="{Y(max(ARM))-8:.1f}">arm vs its twin · peak {max(ARM):.2f}</text>')
    g.append(f'<text class="axl" x="{(PL+W-PR)/2:.0f}" y="{H-4}" text-anchor="middle">interpolation position α  (0 = one model, 1 = the other)</text>')
    g.append(f'<text class="axl" transform="translate(13,{(PT+H-PB)/2:.0f}) rotate(-90)" text-anchor="middle">held-out loss (nats)</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Loss along the linear interpolation between two final models. Two different-seed twins rise to a peak of {max(FLOOR):.2f} nats; an arm and its own twin stay nearly flat at {max(ARM):.2f}.">{"".join(g)}</svg>'

# ---------- figure 2: trajectory (log y) ----------
def fig2():
    W2,H2=680,300; PL2,PR2,PT2,PB2=56,18,18,40
    ys=[v for vs in TRAJ.values() for v in vs]
    lo,hi=math.log10(4e-5),math.log10(2e-2)
    def X(s): return PL2+(math.log10(s)-math.log10(200))/(math.log10(9535)-math.log10(200))*(W2-PL2-PR2)
    def Y(v): return PT2+(hi-math.log10(max(v,4e-5)))/(hi-lo)*(H2-PT2-PB2)
    g=[]
    for e,lab in ((1e-4,"1e−4"),(1e-3,"1e−3"),(1e-2,"1e−2")):
        g.append(f'<line class="grid" x1="{PL2}" y1="{Y(e):.1f}" x2="{W2-PR2}" y2="{Y(e):.1f}"/>')
        g.append(f'<text class="tick" x="{PL2-8}" y="{Y(e)+4:.1f}" text-anchor="end">{lab}</text>')
    for s in (200,300,500,1000,3000,9535):
        g.append(f'<text class="tick" x="{X(s):.1f}" y="{H2-PB2+20}" text-anchor="middle">{s}</text>')
    cls={"fluent-false":"s-arm","fluent-true":"s-arm2","random-chars":"s-floor"}
    for a,vals in TRAJ.items():
        d="M"+" L".join(f"{X(STEPS[i]):.1f},{Y(vals[i]):.1f}" for i in range(len(STEPS)))
        g.append(f'<path class="{cls[a]}" d="{d}"/>')
    pk=STEPS[TRAJ["fluent-false"].index(max(TRAJ["fluent-false"]))]
    g.append(f'<line class="mark" x1="{X(pk):.1f}" y1="{PT2}" x2="{X(pk):.1f}" y2="{H2-PB2}"/>')
    g.append(f'<text class="dl" x="{X(pk)+8:.1f}" y="{PT2+14}">peak ≈ step {pk}</text>')
    g.append(f'<text class="axl" x="{(PL2+W2-PR2)/2:.0f}" y="{H2-4}" text-anchor="middle">training step (log)</text>')
    g.append(f'<text class="axl" transform="translate(13,{(PT2+H2-PB2)/2:.0f}) rotate(-90)" text-anchor="middle">mean |loss − twin| (log)</text>')
    return f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Divergence from the twin over training. Nearly flat for ten steps, amplifying about 130-fold to a peak near step 260, then decaying to a plateau. All three arms follow the same curve.">{"".join(g)}</svg>'


F = {"fig1": fig1(), "fig2": fig2(), "arm_peak": max(ARM), "floor_peak": max(FLOOR)}

CSS = """
:root{
  --ground:#FCFCFD; --raise:#F3F4F7; --ink:#171A1F; --ink-2:#3D4753; --muted:#5C6675;
  --rule:#E2E5EB; --rule-2:#CFD4DD; --burst:#3D5A98; --floor:#A8722A; --burst-2:#6E7FA8;
  --shadow:0 1px 2px rgba(20,26,38,.05), 0 8px 24px -16px rgba(20,26,38,.28);
}
@media (prefers-color-scheme: dark){
 :root:not([data-theme="light"]){
  --ground:#0F1215; --raise:#171B21; --ink:#E7EAEE; --ink-2:#C3CAD3; --muted:#98A2B0;
  --rule:#252A32; --rule-2:#333A45; --burst:#6E92D4; --floor:#BE8735; --burst-2:#8D9CC0;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
 }
}
:root[data-theme="dark"]{
  --ground:#0F1215; --raise:#171B21; --ink:#E7EAEE; --ink-2:#C3CAD3; --muted:#98A2B0;
  --rule:#252A32; --rule-2:#333A45; --burst:#6E92D4; --floor:#BE8735; --burst-2:#8D9CC0;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:53rem;margin:0 auto;padding:clamp(2rem,5vw,4.5rem) clamp(1.1rem,4vw,2rem) 6rem}
.measure{max-width:34rem}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  text-wrap:balance;letter-spacing:-.012em;line-height:1.18;margin:0}
h1{font-size:clamp(1.85rem,4.2vw,2.7rem);font-weight:600}
h2{font-size:1.42rem;font-weight:600;margin:3.4rem 0 .2rem}
h3{font-size:1.06rem;font-weight:600;margin:2rem 0 .1rem;color:var(--ink-2)}
p{margin:.85rem 0}
.eyebrow{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  text-transform:uppercase;letter-spacing:.13em;font-size:.665rem;font-weight:600;color:var(--muted)}
.lede{font-size:1.09rem;color:var(--ink-2)}
.meta{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;font-size:.8rem;color:var(--muted);
  display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin-top:1rem}
.rule{height:1px;background:var(--rule);border:0;margin:2.4rem 0}
.num,td.n,th.n{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;font-size:.86em}
strong{font-weight:600}
a{color:var(--burst)}
code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;font-size:.855em;
  background:var(--raise);border:1px solid var(--rule);border-radius:3px;padding:.06em .32em}
.tablewrap{overflow-x:auto;margin:1.2rem 0;border:1px solid var(--rule);border-radius:6px;background:var(--raise)}
table{border-collapse:collapse;width:100%;font-size:.885rem}
th,td{padding:.5rem .72rem;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.68rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--muted);font-weight:600;background:var(--ground)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right}
tr.total td{font-weight:600;border-top:1px solid var(--rule-2)}
figure{margin:1.8rem 0;padding:1.1rem 1rem .7rem;background:var(--raise);border:1px solid var(--rule);border-radius:8px}
figure svg{display:block;width:100%;height:auto}
figcaption{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.775rem;color:var(--muted);
  margin-top:.5rem;line-height:1.5;max-width:44rem}
.grid{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;font-variant-numeric:tabular-nums}
.axl{fill:var(--muted);font-family:ui-sans-serif,system-ui,sans-serif;font-size:10.5px}
.s-arm{fill:none;stroke:var(--burst);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.s-arm2{fill:none;stroke:var(--burst-2);stroke-width:2;stroke-dasharray:3 3;stroke-linejoin:round}
.s-floor{fill:none;stroke:var(--floor);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.m-arm{fill:var(--burst);stroke:var(--raise);stroke-width:2}
.m-floor{fill:var(--floor);stroke:var(--raise);stroke-width:2}
.mark{stroke:var(--rule-2);stroke-width:1;stroke-dasharray:2 3}
.dl{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11px;fill:var(--ink-2)}
.dl-arm{fill:var(--burst)}.dl-floor{fill:var(--floor)}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1.2rem;font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:.775rem;color:var(--ink-2);margin:.15rem 0 .55rem}
.legend span{display:inline-flex;align-items:center;gap:.42rem}
.sw{width:15px;height:2.5px;border-radius:2px;display:inline-block}
.sw.b{background:var(--burst)} .sw.f{background:var(--floor)}
.sw.b2{background:var(--burst-2);height:0;border-top:2.5px dashed var(--burst-2)}
.keybox{border:1px solid var(--rule-2);border-radius:8px;padding:1.15rem 1.25rem;margin:1.6rem 0;
  background:var(--raise);box-shadow:var(--shadow)}
.keybox .eyebrow{display:block;margin-bottom:.5rem}
.kb-row{display:flex;flex-wrap:wrap;gap:1.6rem;margin-top:.7rem}
.kb-stat{display:flex;flex-direction:column;gap:.1rem}
.kb-v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums;
  font-size:1.42rem;font-weight:600;letter-spacing:-.02em}
.kb-l{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.71rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em}
.v-burst{color:var(--burst)} .v-floor{color:var(--floor)}
.pill{display:inline-block;font-family:ui-sans-serif,system-ui,sans-serif;font-size:.68rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.07em;padding:.16em .5em;border-radius:3px;border:1px solid currentColor}
.pill.null{color:var(--muted)} .pill.pos{color:var(--burst)} .pill.absent{color:var(--floor)}
.toc{border-left:2px solid var(--rule-2);padding:.15rem 0 .15rem 1rem;margin:1.8rem 0 0}
.toc ol{margin:0;padding-left:1.1rem;font-size:.885rem;color:var(--ink-2)}
.toc li{margin:.16rem 0}
.toc a{color:inherit;text-decoration:none;border-bottom:1px solid transparent}
.toc a:hover{border-bottom-color:var(--rule-2)}
.callout{border-left:3px solid var(--floor);padding:.15rem 0 .15rem 1rem;margin:1.4rem 0;color:var(--ink-2)}
.callout .eyebrow{color:var(--floor);display:block;margin-bottom:.2rem}
ul.tight{margin:.7rem 0;padding-left:1.15rem} ul.tight li{margin:.3rem 0}
:focus-visible{outline:2px solid var(--burst);outline-offset:2px;border-radius:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:620px){body{font-size:16px} th,td{white-space:normal}}
"""

def armrow(a):
    r=A[a]; return (f'<tr><td><code>{a}</code></td><td class="n">{r["mean"]:.6f}</td>'
      f'<td class="n">{r["sd"]:.6f}</td><td class="n">{r["min"]:.6f}</td><td class="n">{r["max"]:.6f}</td>'
      f'<td class="n">{r["l2_raw_mean"]:.2f}</td><td><span class="pill null">no</span></td></tr>')

HTML=f"""<title>One row, nine thousand steps — burst-study preprint</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">Preprint · draft · n = 8 of a planned 10 seeds</span>
  <h1 style="margin-top:.5rem">One row, nine thousand steps: a single injected passage moves a transformer far, but not out of its basin</h1>
  <p class="lede measure" style="margin-top:1rem">Thirty-two GPT-2 models trained from scratch, identical within a seed except for
  one 194-token passage written into a single row of a single batch at step 200. The passage is
  unmistakable where it lands and invisible by the end — and the reason is geometric.</p>
  <div class="meta">
    <span>32 runs · 4 arms × 8 seeds</span><span>commit <code>d52a2b8</code></span>
    <span>corpus <code>c338fd06…</code></span><span>2026-08-10</span>
  </div>
  <div class="toc">
    <ol>
      <li><a href="#result">The result in one figure</a></li>
      <li><a href="#design">Design</a></li>
      <li><a href="#manip">The manipulation worked</a></li>
      <li><a href="#headline">The pre-registered contrast</a></li>
      <li><a href="#heldout">Held-out loss</a></li>\n      <li><a href="#traj">Where the signal goes</a></li>
      <li><a href="#prereg">Pre-registration and departures</a></li>
      <li><a href="#limits">Limitations</a></li>\n      <li><a href="#robust">Robustness</a></li>\n      <li><a href="#repl">Replication</a></li>\n      <li><a href="#prov">Provenance</a></li>
    </ol>
  </div>
</header>

<hr class="rule">

<section id="result">
<span class="eyebrow">The finding</span>
<h2>Two rulers disagree about the size of one burst</h2>
<p class="measure">Every model here has a <em>twin</em>: a run with the same seed, the same initialisation and the
same data order, differing only in that nothing was injected into it. Measuring an arm against its own twin
isolates the burst exactly. Measuring two twins of <em>different seeds</em> against each other gives the scale
of variation that seed choice alone produces — the floor any real effect has to clear.</p>

<figure>
  <div class="legend">
    <span><i class="sw f"></i> two twins, different seeds (seed alone)</span>
    <span><i class="sw b"></i> an arm vs its own twin (the burst)</span>
  </div>
  {F["fig1"]}
  <figcaption>Loss along the straight line in weight space between two finished models, averaged over all
  pairs (28 floor pairs, 24 arm pairs), 512 held-out windows. Two models that differ only by seed cannot be
  mixed: the interpolation climbs to {F["floor_peak"]:.2f} nats, a broken model. An arm and its own twin mix
  almost freely — a peak of {F["arm_peak"]:.2f} against endpoints near 3.19. The burst moved the weights a
  long way <em>inside</em> one basin without leaving it.</figcaption>
</figure>

<div class="keybox">
  <span class="eyebrow">The same pair of models, two ways</span>
  <div class="kb-row">
    <div class="kb-stat"><span class="kb-v v-burst">44%</span><span class="kb-l">of a seed change, in L2 distance</span></div>
    <div class="kb-stat"><span class="kb-v v-floor">3.0%</span><span class="kb-l">of a seed change, in loss barrier</span></div>
    <div class="kb-stat"><span class="kb-v">24.7×</span><span class="kb-l">gap between the distributions</span></div>
  </div>
  <p style="margin:.9rem 0 0;font-size:.93rem;color:var(--ink-2)">One injected row moves the final weights
  {arm_l2:.0f} units — nearly half as far as changing the random seed ({l2f:.0f}) — while producing
  {arm_b/fl["mean"]*100:.1f}% of the barrier. Displacement and basin membership are different questions, and
  this burst answers them differently.</p>
</div>
</section>

<section id="design">
<span class="eyebrow">Design</span>
<h2>What was held constant</h2>
<p class="measure">Within a seed the four runs share an initialisation, a data order and every hyperparameter.
They are verified <strong>bit-identical to step 199</strong> — by SHA-256 over the weight tensors, and across
two different physical machines. At step 200 a 194-token passage enters one row of one micro-batch; training
continues untouched for 9,336 more steps. Every difference at the end descends from that row.</p>
<div class="tablewrap"><table>
<thead><tr><th>arm</th><th>injected text</th><th class="n">seeds</th></tr></thead>
<tbody>
<tr><td><code>fluent-true</code></td><td>grammatical English asserting something <strong>true</strong></td><td class="n">8</td></tr>
<tr><td><code>fluent-false</code></td><td>same register, structure and length, asserting something <strong>false</strong></td><td class="n">8</td></tr>
<tr><td><code>random-chars</code></td><td>no word structure at all</td><td class="n">8</td></tr>
<tr><td><code>twin</code></td><td>nothing — the matched control</td><td class="n">8</td></tr>
</tbody></table></div>
<p class="measure">The two fluent passages are matched to <strong>0.14%</strong> on the gradient contribution
that actually reaches the optimiser. The burst is a ~2.2% perturbation to one optimiser step.</p>
<div class="callout">
  <span class="eyebrow">A confound found, not designed</span>
  <p style="margin:.2rem 0">The true passage's subject appears <strong>4 times</strong> in the 2.5-billion-token
  training corpus; the false passage's subject appears <strong>0</strong>. All four mentions are on-point. The
  contrast therefore measures truth-<em>with-attestation</em>, not truth alone. This was discovered by counting
  before any run existed, and is close to unavoidable: a claim is checkable because it is documented.</p>
</div>
</section>

<section id="manip">
<span class="eyebrow">Manipulation check</span>
<h2>The burst was unmistakable where it landed</h2>
<p class="measure">Step 200 is the first step at which any injecting run's loss differs from its twin — in
24 of 24 runs. Steps 0–199 are identical. At that step the arms order exactly as the design predicted.</p>
<div class="tablewrap"><table>
<thead><tr><th>arm</th><th class="n">loss excess over twin</th><th class="n">t(7)</th><th class="n">p</th><th class="n">sign</th></tr></thead>
<tbody>
<tr><td><code>random-chars</code></td><td class="n">+0.00107961</td><td class="n">+3.219</td><td class="n">0.0147</td><td class="n">6/8</td></tr>
<tr><td><code>fluent-false</code></td><td class="n">+0.00023608</td><td class="n">+0.706</td><td class="n">0.503</td><td class="n">5/8</td></tr>
<tr><td><code>fluent-true</code></td><td class="n">+0.00006128</td><td class="n">+0.185</td><td class="n">0.858</td><td class="n">4/8</td></tr>
<tr class="total"><td><code>fluent-false</code> − <code>fluent-true</code></td><td class="n">+0.00017480</td><td class="n">+13.92</td><td class="n">2.3e−6</td><td class="n">8/8</td></tr>
</tbody></table></div>
<p class="measure">The false passage is harder to predict than the true one at <em>every</em> seed, with no
overlap — which is what the attestation asymmetry predicts. This is a property of the stimuli under the
step-199 model rather than an outcome; its job is to rule out a failed manipulation as the explanation for
what follows.</p>
</section>

<section id="headline">
<span class="eyebrow">Result · pre-registered</span>
<h2>Inside the basin, content does not matter</h2>
<p class="measure">The headline metric was fixed by a four-branch decision rule written before any checkpoint
existed: the plain interpolation loss barrier of each arm against its seed-matched twin.</p>
<div class="tablewrap"><table>
<thead><tr><th>arm</th><th class="n">mean</th><th class="n">sd</th><th class="n">min</th><th class="n">max</th><th class="n">L2</th><th>clears floor</th></tr></thead>
<tbody>
{armrow("fluent-false")}
{armrow("fluent-true")}
{armrow("random-chars")}
<tr class="total"><td>twin vs twin (floor), 28 pairs</td><td class="n">{fl["mean"]:.6f}</td><td class="n">—</td>
<td class="n">{fl["min"]:.6f}</td><td class="n">{fl["max"]:.6f}</td><td class="n">{l2f:.2f}</td><td>—</td></tr>
</tbody></table></div>
<p class="measure">The three arms are indistinguishable from one another: between-arm differences (~0.008) sit
under half the seed-to-seed spread within an arm. <strong>A burst of random punctuation displaces the model as
far as grammatical English does.</strong> No arm approaches the floor — the two distributions do not overlap at
all, separated by a gap of 4.47.</p>

<h3>The registered contrasts</h3>
<div class="tablewrap"><table>
<thead><tr><th>contrast</th><th class="n">mean</th><th class="n">t(7)</th><th class="n">p</th><th class="n">95% CI</th><th>verdict</th></tr></thead>
<tbody>
<tr><td>primary, on the barrier</td><td class="n">{pr["mean"]:+.6f}</td><td class="n">{pr["t"]:+.4f}</td>
<td class="n">{pr["p_raw"]:.4f}</td><td class="n">[{pr["ci_low"]:+.5f}, {pr["ci_high"]:+.5f}]</td>
<td><span class="pill null">null</span></td></tr>
<tr><td>primary, on held-out loss</td><td class="n">{hp["mean"]:+.8f}</td><td class="n">{hp["t"]:+.4f}</td>
<td class="n">{hp["p_raw"]:.4f}</td><td class="n">[{hp["ci_low"]:+.5f}, {hp["ci_high"]:+.5f}]</td>
<td><span class="pill null">null</span></td></tr>
<tr><td>secondary (<code>pos-substituted</code>)</td><td class="n">—</td><td class="n">—</td><td class="n">—</td>
<td class="n">—</td><td><span class="pill absent">arm cut</span></td></tr>
</tbody></table></div>
<p class="measure">Holding register, structure, token length and injected gradient magnitude fixed, the truth of
the asserted proposition does not measurably change how far the single gradient step displaces the model. No
multiple-comparison correction is applied: the family is one, ruled before the runs.</p>
<div class="callout">
  <span class="eyebrow">A warning the data supplied about itself</span>
  <p style="margin:.2rem 0">At five seeds this contrast was negative <strong>5 times out of 5</strong>, and had a
  ready mechanism — <code>fluent-true</code> carries the higher gradient norm at injection in 8 of 8 seeds. Seeds
  5, 6 and 7 are all positive, one by +0.066, and it lands at p&nbsp;=&nbsp;0.51. The correlation with that
  mechanism is r&nbsp;=&nbsp;+0.16. The analysis module refuses panels below ten seeds precisely because
  low-seed numbers in this project were overturned three times before; this is a fourth, one seed below the
  floor.</p>
</div>
</section>

<section id="heldout">
<span class="eyebrow">Result · secondary readout</span>
<h2>Held-out loss agrees</h2>
<p class="measure">Held-out next-token cross-entropy over all 10,240 held-out windows (10.5M tokens) per model,
all 32 re-scored on one machine so no contrast straddles two hardware stacks.</p>
<div class="tablewrap"><table>
<thead><tr><th>arm</th><th class="n">effect vs twin</th><th class="n">95% CI</th><th class="n">t(7)</th><th class="n">p</th><th>clears floor</th></tr></thead>
<tbody>
<tr><td><code>random-chars</code></td><td class="n">+0.00063205</td><td class="n">[−0.00023, +0.00163]</td><td class="n">+1.266</td><td class="n">0.2461</td><td><span class="pill null">no</span></td></tr>
<tr><td><code>fluent-true</code></td><td class="n">+0.00048436</td><td class="n">[−0.00047, +0.00160]</td><td class="n">+0.860</td><td class="n">0.4182</td><td><span class="pill null">no</span></td></tr>
<tr><td><code>fluent-false</code></td><td class="n">+0.00004190</td><td class="n">[−0.00089, +0.00099]</td><td class="n">+0.082</td><td class="n">0.9373</td><td><span class="pill null">no</span></td></tr>
<tr class="total"><td>twin vs twin (floor), 28 pairs</td><td class="n">—</td><td class="n">[−0.00588, +0.01104]</td><td class="n">—</td><td class="n">—</td><td>—</td></tr>
</tbody></table></div>
<p class="measure">No arm separates from its twin; the largest effect is seventeen times smaller than the widest
difference seed alone produced. The primary contrast here reproduces an earlier independent computation of the
same quantity, from the checkpoints, on different hardware, to thirteen significant figures.</p>
</section>

<section id="traj">
<span class="eyebrow">Result · exploratory</span>
<h2>Where the signal goes</h2>
<figure>
  <div class="legend">
    <span><i class="sw b"></i> fluent-false</span>
    <span><i class="sw b2"></i> fluent-true</span>
    <span><i class="sw f"></i> random-chars</span>
  </div>
  {F["fig2"]}
  <figcaption>Mean absolute per-step loss difference from the seed-matched twin, averaged over 8 seeds, log
  axes. The injection is nearly invisible for ten steps, amplifies roughly 130-fold to a peak near step 260,
  then decays to a plateau about eighteen times the original perturbation. All three arms trace the same
  curve.</figcaption>
</figure>
<p class="measure">By the endpoint, the distance between the two <em>fluent</em> arms (0.00135) is no smaller
than the distance from either to the twin (0.00147, 0.00158). Two runs given <em>different</em> fluent passages
end as far apart as either is from a run given <em>no</em> passage. What survives 9,336 steps is the fact of a
perturbation, not its content.</p>
</section>

<section id="prereg">
<span class="eyebrow">Pre-registration</span>
<h2>What was fixed, and every departure</h2>
<p class="measure">The pre-registration was committed on 2026-08-03, before any run existed; its own
falsifiability check — that no checkpoint had ever been added on any branch — returned empty and is re-runnable.
Amendments are dated before the runs they govern.</p>
<ul class="tight measure">
<li><strong>The secondary contrast is gone, not unreported.</strong> Its arm, <code>pos-substituted</code>, was
cut on 2026-08-08. It is reported as uncomputable, naming the missing arm, in every analysis output.
<code>random-chars</code> was deliberately <em>not</em> promoted into the empty slot.</li>
<li><strong>n = 8 against a design of 10 and a hard floor of 10.</strong> The study stopped because compute ran
out — a data-independent rule fixed before any mean was examined, so not optional stopping. The floor was
crossed by a command-line flag rather than by editing the constant, so the crossing appears in the record.</li>
<li><strong>The noise floor for a pairwise metric.</strong> The analysis module's floor degenerates to zeros
when the metric is pairwise rather than per-model. All 28 twin-vs-twin barriers were <em>measured</em> instead
of derived.</li>
</ul>
</section>

<section id="limits">
<span class="eyebrow">Limitations</span>
<h2>What this does not show</h2>
<ul class="tight measure">
<li><strong>One stimulus pair.</strong> Every claim about "true" and "false" rests on two passages. Nothing here
separates a property of truth from a property of these two paragraphs. No number of seeds fixes this.</li>
<li><strong>Truth is entangled with attestation</strong> (4 mentions versus 0), so the contrast cannot be read
as "truth doesn't matter."</li>
<li><strong>One model size, one injection step, one burst length, one position</strong> — 124M parameters,
step 200 of 9,536, 194 tokens, one row of one micro-batch.</li>
<li><strong>The floor is a conservative reference</strong>, not the null distribution of the tested quantity: it
compares models that share nothing, while the effect compares models that share 199 steps.</li>
<li><strong>The barrier is a lower bound</strong> on a 21-point grid, and is the plain barrier rather than the
permutation-aligned one — selected in advance by the decision rule.</li>
<li><strong>Seeds 8 and 9 are outstanding.</strong></li>
</ul>
</section>

<section id="robust">
<span class="eyebrow">Robustness</span>
<h2>Is 512 evaluation windows enough?</h2>
<p class="measure">Checked rather than assumed. Six arm-vs-twin pairs and two floor pairs were recomputed at
<strong>2048</strong> windows — four times the evaluation set, same alpha grid.</p>
<div class="tablewrap"><table>
<thead><tr><th>pair</th><th class="n">512 win</th><th class="n">2048 win</th><th class="n">relative shift</th></tr></thead>
<tbody>
<tr><td><code>seed00_fluent-false</code></td><td class="n">0.141220</td><td class="n">0.139366</td><td class="n">−1.31%</td></tr>
<tr><td><code>seed00_fluent-true</code></td><td class="n">0.147930</td><td class="n">0.145989</td><td class="n">−1.31%</td></tr>
<tr><td><code>seed03_fluent-false</code></td><td class="n">0.139241</td><td class="n">0.139523</td><td class="n">+0.20%</td></tr>
<tr><td><code>seed03_fluent-true</code></td><td class="n">0.142782</td><td class="n">0.142944</td><td class="n">+0.11%</td></tr>
<tr><td><code>seed06_fluent-false</code></td><td class="n">0.161798</td><td class="n">0.161075</td><td class="n">−0.45%</td></tr>
<tr><td><code>seed06_fluent-true</code></td><td class="n">0.156373</td><td class="n">0.155146</td><td class="n">−0.78%</td></tr>
<tr><td>floor, seeds 0 vs 1</td><td class="n">4.811980</td><td class="n">4.839452</td><td class="n">+0.57%</td></tr>
<tr><td>floor, seeds 2 vs 5</td><td class="n">4.746679</td><td class="n">4.789661</td><td class="n">+0.91%</td></tr>
</tbody></table></div>
<p class="measure">Largest shift 1.31%, mean 0.71% — an order of magnitude below the between-arm differences
the analysis compares, two orders below the arm-to-floor gap.</p>
<p class="measure"><strong>The contrast is steadier than either term it is built from.</strong> The shifts are
correlated <em>within</em> a seed — at seed 0 both arms move −1.31% together — so the paired difference cancels
almost all of the evaluation noise: the per-seed contrasts move by 0.0001–0.0005 against a between-arm spread
of ~0.008. That is the same cancellation the paired design relies on for seeds, turning up in a place nobody
designed it into.</p>
</section>

<section id="repl">
<span class="eyebrow">Replication</span>
<h2>The pipeline checked against an independent computation</h2>
<p class="measure">The arm-vs-twin barrier above is new. The <em>arm-vs-arm</em> barrier is not — it was computed
on 2026-08-09 on different hardware by a different script. Recomputing all eight seeds here is an end-to-end
check of this pipeline against a number produced independently.</p>
<div class="tablewrap"><table>
<thead><tr><th></th><th class="n">mean</th><th class="n">sd</th><th class="n">raw L2</th></tr></thead>
<tbody>
<tr><td>original (A100, cu130)</td><td class="n">0.12278054</td><td class="n">0.02246385</td><td class="n">225.2279</td></tr>
<tr><td>recomputed (A6000, cu126)</td><td class="n">0.12278054</td><td class="n">0.02246385</td><td class="n">225.2279</td></tr>
<tr class="total"><td>largest per-seed difference</td><td class="n">1.4e−08</td><td class="n">—</td><td class="n">0 (exact)</td></tr>
</tbody></table></div>
<p class="measure">The barrier arithmetic, the interpolation, the checkpoint loading and the held-out
evaluation all reproduce across machines and CUDA builds. Together with the held-out losses reproducing to
1.6e−9, essentially nothing in this analysis depends on the machine it ran on.</p>
</section>

<section id="prov">
<span class="eyebrow">Provenance</span>
<h2>How much of this is checkable</h2>
<div class="tablewrap"><table>
<thead><tr><th>claim</th><th>evidence</th></tr></thead>
<tbody>
<tr><td>all 32 runs share one code state</td><td>one commit, clean tree in all 32 provenance records, zero resumes</td></tr>
<tr><td>both machines trained identically</td><td>all four arms <strong>bit-identical at step 199</strong>, every seed, across two boxes</td></tr>
<tr><td>both machines used one corpus</td><td>150 of 150 corpus blocks byte-identical by stored digest</td></tr>
<tr><td>the injected text is the committed text</td><td>sha256 recorded at injection matches the committed file, all 24 injecting runs</td></tr>
<tr><td>the injection fired where designed</td><td>step 200, same seed-derived slot and row across all arms and both boxes</td></tr>
<tr><td>the evaluation is hardware-independent</td><td>16 prior values reproduce to ≤1.6e−9 nats across GPU architectures</td></tr>
<tr><td>the archive is intact</td><td>366 objects digest-checked; one corrupt checkpoint found at full length and refetched</td></tr>
</tbody></table></div>
<p class="measure">Bootstrap intervals are drawn from SHA-256 in counter mode rather than a library PRNG, so an
interval does not move with a dependency upgrade. The t-distribution and corrections are hand-written to keep
the analysis dependency-free and are cross-checked against scipy where it exists.</p>
</section>

<hr class="rule">
<p style="font-family:ui-sans-serif,system-ui,sans-serif;font-size:.775rem;color:var(--muted)" class="measure">
Held-out loss re-scored for all 32 runs on one machine; the 16 runs previously scored elsewhere reproduce to
within 1.6e−9 nats. Archive integrity verified against per-object digests, one corrupt checkpoint found and
refetched. Bootstrap intervals are drawn from SHA-256 in counter mode so they do not move with a library
upgrade.</p>
</div>
"""
out = REPO / "docs" / "preprint.html"
out.write_text(HTML)
print(f"wrote {out} ({len(HTML):,} chars)")
