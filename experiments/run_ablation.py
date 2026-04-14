"""
Ablation study for SemiFA Root Cause Analyzer — Paper Table 4.

Tests Agent 2 (Root Cause Analyzer) under 4 conditions:
  1. Full SemiFA  — Qdrant retrieval + equipment telemetry logs
  2. No Retrieval — equipment telemetry only (no Qdrant)
  3. No Telemetry — Qdrant retrieval only (no logs)
  4. Baseline     — image description only (no retrieval, no logs)

Each condition's output is scored by GPT-4o as an automated expert judge
on three criteria (1–5 scale each):
  - Specificity      : Is the root cause precise and semiconductor-domain-specific?
  - Actionability    : Does it give concrete corrective actions?
  - Grounding        : Is it grounded in the provided evidence (logs / history)?

Results → experiments/ablation_results.json + paper-ready Table 4 to stdout.

Requires:
    pip install openai

Usage:
    python experiments/run_ablation.py --api-key sk-...
    python experiments/run_ablation.py   # if OPENAI_API_KEY env var is set
    python experiments/run_ablation.py --cases 3 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import openai


# ── Prompt templates ──────────────────────────────────────────────────────────

ROOT_CAUSE_PROMPT = """\
You are a semiconductor failure analysis expert.

DEFECT DESCRIPTION:
{description}

DEFECT CLASS: {defect_class} (confidence: {confidence})

RECENT EQUIPMENT LOGS (last 2 hours):
{equipment_logs}

HISTORICALLY SIMILAR DEFECTS:
{similar_defects}

Based on all evidence above, generate:
1. Top 3 ranked root cause hypotheses (most likely first)
2. Supporting evidence for each hypothesis
3. A concise root cause summary (2-3 sentences)

Format your response as:
HYPOTHESIS 1: <cause>
EVIDENCE: <supporting evidence>

HYPOTHESIS 2: <cause>
EVIDENCE: <supporting evidence>

HYPOTHESIS 3: <cause>
EVIDENCE: <supporting evidence>

ROOT CAUSE SUMMARY: <concise summary>
"""

JUDGE_PROMPT = """\
You are a senior semiconductor process engineer evaluating an AI-generated \
failure analysis root cause report.

Score the following root cause output on three criteria (integer 1-5 each):

SCORING RUBRIC
--------------
Specificity (1-5):
  5 = Cites exact equipment parameters, process step names, defect mechanism by name
  4 = References specific process stages or measurement values
  3 = Mentions process category (e.g. "lithography") without specifics
  2 = Vague domain language (e.g. "contamination issue")
  1 = Generic or off-topic text

Actionability (1-5):
  5 = Prescribes concrete corrective steps (e.g. "reduce chuck temperature by 2 C")
  4 = Clear direction with partial specifics
  3 = General category of fix (e.g. "inspect edge handling")
  2 = Only diagnostic suggestion, no fix
  1 = No actionable content

Grounding (1-5):
  5 = Every claim traceable to provided evidence (logs / similar defects)
  4 = Most claims grounded, minor inference
  3 = Mix of grounded and hallucinated claims
  2 = Mostly hallucinated or ignores provided evidence
  1 = Completely ignores evidence

CONTEXT PROVIDED TO THE AI
---------------------------
Defect class : {defect_class}
Equipment logs available : {has_logs}
Historical similar defects available : {has_similar}

AI OUTPUT TO SCORE
------------------
{ai_output}

Respond ONLY with valid JSON in exactly this format (no markdown, no code fences):
{{"specificity": <1-5>, "actionability": <1-5>, "grounding": <1-5>, "rationale": "<one sentence>"}}
"""


# ── Synthetic test cases ──────────────────────────────────────────────────────
# Equipment logs are formatted as SECS-II S5F1 (Alarm Report) and S6F11
# (Collection Event Report) messages per SEMI E5/E37, as they would appear
# in a host-side HSMS log.  Fields:
#   S5F1  — AlarmID (hex), ALCD (0x81=set), AlarmCode, AlarmText, affected DVID
#   S6F11 — DATAID, CEID (Collection Event ID + mnemonic), RPTID, DVID values
# These are representative logs manually authored for the ablation study;
# a live deployment would ingest identical message structures via HSMS.

TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "TC-01",
        "defect_class": "scratch",
        "confidence": "94.2%",
        "description": (
            "A linear dark track approximately 2 mm long is visible crossing "
            "the wafer surface at 45 degrees. Track width is ~5 um, consistent "
            "with a single contact event. Die yield along the track is 0%."
        ),
        "equipment_logs": [
            "[2024-03-15 08:10:00.003] <- S6F11 DATAID=1 CEID=1001(WAFER_LOAD_START)"
            " RPT[RPTID=101]: DVID:wafer_id=W03 DVID:lot_id=LOT-2024-031 DVID:slot=03",
            "[2024-03-15 08:11:28.774] <- S5F1 AlarmID=0x0031 ALCD=0x81"
            " AlarmCode=VAC_LOW AlarmText=\"End-effector vacuum below LCL\""
            " DVID:end_effector_vacuum=55mbar LCL=65mbar",
            "[2024-03-15 08:11:44.912] <- S6F11 DATAID=2 CEID=1004(PROCESS_STATE_CHANGE)"
            " RPT[RPTID=102]: DVID:wafer_handler_speed=0.45m/s"
            " DVID:prev_state=TRANSFERRING DVID:curr_state=IDLE",
            "[2024-03-15 08:12:01.238] <- S5F1 AlarmID=0x0066 ALCD=0x81"
            " AlarmCode=PRESSURE_HIGH AlarmText=\"Chuck clamp pressure exceeded UCL\""
            " DVID:chuck_pressure=2.8bar UCL=2.5bar",
        ],
        "similar_defects": [
            {"score": 0.93, "defect_type": "scratch",
             "root_cause": "End-effector contact during wafer transfer — vacuum loss event"},
            {"score": 0.88, "defect_type": "scratch",
             "root_cause": "Chuck particle contamination causing mechanical contact"},
        ],
    },
    {
        "id": "TC-02",
        "defect_class": "edge_crack",
        "confidence": "91.7%",
        "description": (
            "Branching crack initiating from the wafer edge at 3 o'clock "
            "position, propagating ~8 mm inward. SEM imaging shows transgranular "
            "fracture morphology. Adjacent die show delamination stress patterns."
        ),
        "equipment_logs": [
            "[2024-03-15 09:29:18.001] <- S6F11 DATAID=5 CEID=2001(PROCESS_START)"
            " RPT[RPTID=201]: DVID:wafer_id=W07 DVID:lot_id=LOT-2024-038"
            " DVID:recipe_id=DICE_200MM_V2",
            "[2024-03-15 09:29:38.447] <- S5F1 AlarmID=0x00A3 ALCD=0x81"
            " AlarmCode=BLADE_WEAR AlarmText=\"Blade wear counter exceeded PM threshold\""
            " DVID:blade_wear_counter=42150 PM_threshold=40000",
            "[2024-03-15 09:29:53.882] <- S5F1 AlarmID=0x0055 ALCD=0x81"
            " AlarmCode=COOLANT_FLOW_LOW AlarmText=\"Coolant flow rate below LCL\""
            " DVID:coolant_flow_rate=0.8L/min LCL=1.2L/min",
            "[2024-03-15 09:30:09.115] <- S6F11 DATAID=6 CEID=2004(PROCESS_STATE_CHANGE)"
            " RPT[RPTID=202]: DVID:dicing_blade_rpm=28500"
            " DVID:prev_state=CUTTING DVID:curr_state=PAUSED",
        ],
        "similar_defects": [
            {"score": 0.91, "defect_type": "edge_crack",
             "root_cause": "Worn dicing blade combined with insufficient coolant — thermal stress fracture"},
            {"score": 0.84, "defect_type": "edge_crack",
             "root_cause": "Excessive blade speed causing chip-out at wafer edge"},
        ],
    },
    {
        "id": "TC-03",
        "defect_class": "particle_contamination",
        "confidence": "87.5%",
        "description": (
            "Bright spots of 0.3-1.2 um diameter distributed across the active "
            "die area. EDS analysis suggests silicon oxide particles. Distribution "
            "is non-uniform, denser near the wafer center."
        ),
        "equipment_logs": [
            "[2024-03-15 10:14:08.553] <- S6F11 DATAID=9 CEID=3001(PROCESS_START)"
            " RPT[RPTID=301]: DVID:wafer_id=W11 DVID:lot_id=LOT-2024-044"
            " DVID:recipe_id=CVD_SIO2_400NM",
            "[2024-03-15 10:14:39.220] <- S5F1 AlarmID=0x00C1 ALCD=0x81"
            " AlarmCode=CLEAN_OVERDUE AlarmText=\"Chamber dry-clean cycle count exceeded limit\""
            " DVID:chamber_clean_cycles=187 limit=150",
            "[2024-03-15 10:14:53.771] <- S6F11 DATAID=10 CEID=3002(PROCESS_DATA_REPORT)"
            " RPT[RPTID=302]: DVID:susceptor_temp=650C DVID:purge_gas_flow=12.5sccm",
            "[2024-03-15 10:15:19.644] <- S5F1 AlarmID=0x0078 ALCD=0x81"
            " AlarmCode=PRESSURE_SPIKE AlarmText=\"CVD chamber pressure transient above UCL\""
            " DVID:cvd_chamber_pressure=4.2mTorr UCL=3.5mTorr",
        ],
        "similar_defects": [
            {"score": 0.89, "defect_type": "particle_contamination",
             "root_cause": "CVD chamber wall deposition flaking — clean cycle overdue"},
            {"score": 0.82, "defect_type": "particle_contamination",
             "root_cause": "Susceptor coating degradation generating SiO2 particulates"},
        ],
    },
    {
        "id": "TC-04",
        "defect_class": "ring_pattern",
        "confidence": "96.1%",
        "description": (
            "Concentric ring of defective dies at 60-70% radius from wafer center. "
            "Good die distribution outside and inside the ring. Pattern matches "
            "spin-coat non-uniformity signature."
        ),
        "equipment_logs": [
            "[2024-03-15 11:04:28.002] <- S6F11 DATAID=13 CEID=4001(PROCESS_START)"
            " RPT[RPTID=401]: DVID:wafer_id=W15 DVID:lot_id=LOT-2024-051"
            " DVID:recipe_id=SPIN_COAT_PR_248NM",
            "[2024-03-15 11:04:49.338] <- S5F1 AlarmID=0x00D4 ALCD=0x81"
            " AlarmCode=HUMIDITY_HIGH AlarmText=\"Track ambient humidity exceeded UCL\""
            " DVID:ambient_humidity=62pct UCL=45pct",
            "[2024-03-15 11:05:14.667] <- S5F1 AlarmID=0x00B2 ALCD=0x81"
            " AlarmCode=SPEED_UNSTABLE AlarmText=\"Spin chuck RPM deviation exceeded tolerance\""
            " DVID:spin_speed=2200rpm setpoint=2000rpm tolerance=50rpm",
            "[2024-03-15 11:05:29.891] <- S6F11 DATAID=14 CEID=4002(PROCESS_DATA_REPORT)"
            " RPT[RPTID=402]: DVID:resist_volume=1.8mL DVID:exhaust_flow=145CFM",
        ],
        "similar_defects": [
            {"score": 0.95, "defect_type": "ring_pattern",
             "root_cause": "Spin coater speed instability causing resist pile-up at resonance radius"},
            {"score": 0.87, "defect_type": "ring_pattern",
             "root_cause": "High ambient humidity causing solvent evaporation non-uniformity during spin"},
        ],
    },
    {
        "id": "TC-05",
        "defect_class": "center_cluster",
        "confidence": "88.9%",
        "description": (
            "Cluster of defective dies concentrated within 20 mm of wafer center. "
            "Pattern suggests CMP over-polishing at center. Affected dies show "
            "metal layer thinning under SEM cross-section."
        ),
        "equipment_logs": [
            "[2024-03-15 12:19:17.004] <- S6F11 DATAID=17 CEID=5001(PROCESS_START)"
            " RPT[RPTID=501]: DVID:wafer_id=W22 DVID:lot_id=LOT-2024-059"
            " DVID:recipe_id=CMP_ILD_STEP1",
            "[2024-03-15 12:19:38.553] <- S5F1 AlarmID=0x00E7 ALCD=0x81"
            " AlarmCode=PAD_WORN AlarmText=\"CMP pad condition index below minimum\""
            " DVID:pad_condition_index=0.62 minimum=0.75",
            "[2024-03-15 12:19:54.219] <- S6F11 DATAID=18 CEID=5002(PROCESS_DATA_REPORT)"
            " RPT[RPTID=502]: DVID:platen_speed=87rpm DVID:slurry_flow=280mL/min",
            "[2024-03-15 12:20:09.882] <- S5F1 AlarmID=0x00F3 ALCD=0x81"
            " AlarmCode=DOWNFORCE_HIGH AlarmText=\"CMP carrier downforce exceeded UCL\""
            " DVID:cmp_downforce=3.1psi UCL=2.8psi",
        ],
        "similar_defects": [
            {"score": 0.92, "defect_type": "center_cluster",
             "root_cause": "CMP excessive downforce with worn pad causing center over-polish"},
            {"score": 0.85, "defect_type": "center_cluster",
             "root_cause": "Non-uniform slurry distribution — center starvation leading to scratching"},
        ],
    },
]


# ── Condition builders ────────────────────────────────────────────────────────

def _fmt_logs(logs: list[str]) -> str:
    return "\n".join(f"  {l}" for l in logs) if logs else "No equipment logs available."


def _fmt_similar(similar: list[dict]) -> str:
    if not similar:
        return "No historical matches found."
    return "\n".join(
        f"  {i}. [{s['score']:.2f}] {s['defect_type']} — Root cause: {s['root_cause']}"
        for i, s in enumerate(similar, 1)
    )


def build_prompt(case: dict[str, Any], condition: str) -> tuple[str, bool, bool]:
    """Return (prompt, has_logs, has_similar) for the given condition key."""
    has_logs    = condition in ("full", "no_retrieval")
    has_similar = condition in ("full", "no_telemetry")

    prompt = ROOT_CAUSE_PROMPT.format(
        description=case["description"],
        defect_class=case["defect_class"],
        confidence=case["confidence"],
        equipment_logs=_fmt_logs(case["equipment_logs"]) if has_logs else "No equipment logs available.",
        similar_defects=_fmt_similar(case["similar_defects"]) if has_similar else "No historical matches found.",
    )
    return prompt, has_logs, has_similar


# ── OpenAI calls ──────────────────────────────────────────────────────────────

def generate_root_cause(client: openai.OpenAI, model: str, prompt: str) -> str:
    """Generate root cause hypothesis using OpenAI model."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def judge_output(
    client: openai.OpenAI,
    model: str,
    ai_output: str,
    defect_class: str,
    has_logs: bool,
    has_similar: bool,
) -> dict[str, Any]:
    """Score the root-cause output using OpenAI model as judge."""
    prompt = JUDGE_PROMPT.format(
        defect_class=defect_class,
        has_logs="yes" if has_logs else "no",
        has_similar="yes" if has_similar else "no",
        ai_output=ai_output,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Judge returned non-JSON: {raw[:200]}")


# ── Experiment runner ─────────────────────────────────────────────────────────

CONDITIONS = [
    ("full",          "Full SemiFA (Retrieval + Telemetry)"),
    ("no_retrieval",  "No Retrieval (Telemetry only)"),
    ("no_telemetry",  "No Telemetry (Retrieval only)"),
    ("baseline",      "Baseline (Description only)"),
]


def run_ablation(
    client: openai.OpenAI,
    model: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for case in cases:
        print(f"\n{'─' * 60}")
        print(f"Case {case['id']} — {case['defect_class']}")
        print(f"{'─' * 60}")
        case_results = {
            "case_id": case["id"],
            "defect_class": case["defect_class"],
            "conditions": {},
        }

        for cond_key, cond_label in CONDITIONS:
            print(f"  [{cond_label}] ", end="", flush=True)
            t0 = time.time()

            prompt, has_logs, has_similar = build_prompt(case, cond_key)
            ai_output = generate_root_cause(client, model, prompt)
            scores = judge_output(client, model, ai_output, case["defect_class"], has_logs, has_similar)

            elapsed = time.time() - t0
            composite = (scores["specificity"] + scores["actionability"] + scores["grounding"]) / 3

            print(
                f"spec={scores['specificity']}  act={scores['actionability']}  "
                f"gnd={scores['grounding']}  avg={composite:.2f}  ({elapsed:.1f}s)"
            )

            case_results["conditions"][cond_key] = {
                "label": cond_label,
                "ai_output": ai_output,
                "specificity": scores["specificity"],
                "actionability": scores["actionability"],
                "grounding": scores["grounding"],
                "composite": round(composite, 2),
                "rationale": scores.get("rationale", ""),
                "elapsed_s": round(elapsed, 1),
            }

        results.append(case_results)

    return results


# ── Aggregate + print table ───────────────────────────────────────────────────

def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, list[float]]] = {
        k: {"specificity": [], "actionability": [], "grounding": [], "composite": []}
        for k, _ in CONDITIONS
    }
    for case_res in results:
        for cond_key, _ in CONDITIONS:
            cond = case_res["conditions"].get(cond_key, {})
            if cond:
                for metric in ("specificity", "actionability", "grounding", "composite"):
                    agg[cond_key][metric].append(cond[metric])
    return {
        k: {m: round(sum(v) / len(v), 2) for m, v in metrics.items() if v}
        for k, metrics in agg.items()
    }


def print_table4(aggregated: dict[str, dict[str, float]], model: str) -> None:
    print("\n\n" + "=" * 72)
    print(f"TABLE 4 — ABLATION STUDY: ROOT CAUSE REASONING ({model} judge)")
    print("Mean scores across test cases  (scale: 1-5)")
    print("=" * 72)
    print(f"  {'Condition':<38}  {'Spec':>5}  {'Act':>5}  {'Gnd':>5}  {'Avg':>5}")
    print("-" * 72)
    for cond_key, cond_label in CONDITIONS:
        m = aggregated.get(cond_key, {})
        marker = " <-- proposed" if cond_key == "full" else ""
        print(
            f"  {cond_label:<38}  {m.get('specificity',0):>5.2f}  "
            f"{m.get('actionability',0):>5.2f}  {m.get('grounding',0):>5.2f}  "
            f"{m.get('composite',0):>5.2f}{marker}"
        )
    print("-" * 72)

    full = aggregated.get("full", {})
    base = aggregated.get("baseline", {})
    if full and base:
        delta = full.get("composite", 0) - base.get("composite", 0)
        print(f"\n  Full SemiFA vs Baseline: +{delta:.2f} avg points improvement")

    print("\n\nLaTeX table (copy into paper):\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study: root cause reasoning quality scored by " + model + r".}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lcccr}")
    print(r"\toprule")
    print(r"Condition & Specificity & Actionability & Grounding & Avg \\")
    print(r"\midrule")
    for cond_key, cond_label in CONDITIONS:
        m = aggregated.get(cond_key, {})
        b0 = r"\textbf{" if cond_key == "full" else ""
        b1 = r"}" if cond_key == "full" else ""
        print(
            f"{b0}{cond_label}{b1} & {m.get('specificity',0):.2f} & "
            f"{m.get('actionability',0):.2f} & {m.get('grounding',0):.2f} & "
            f"{m.get('composite',0):.2f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    n_cases: int = 5,
    save_results: bool = True,
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> None:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "ERROR: OpenAI API key not found.\n"
            "  Pass it with:  --api-key sk-...\n"
            "  Or set env var: set OPENAI_API_KEY=sk-..."
        )

    client = openai.OpenAI(api_key=key)
    cases = TEST_CASES[:n_cases]

    print(f"SemiFA Ablation Study")
    print(f"Model (generator + judge) : {model}")
    print(f"Test cases   : {len(cases)}")
    print(f"Conditions   : {len(CONDITIONS)}")
    print(f"Total API calls : {len(cases) * len(CONDITIONS) * 2}  (generate + judge per condition)")

    t_start = time.time()
    results = run_ablation(client, model, cases)
    elapsed_total = time.time() - t_start

    aggregated = aggregate(results)
    print_table4(aggregated, model)
    print(f"\nTotal experiment time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    if save_results:
        out = {
            "experiment": "SemiFA ablation study — Root Cause Analyzer",
            "judge_model": model,
            "generator_model": model,
            "n_cases": len(cases),
            "conditions": [{"key": k, "label": l} for k, l in CONDITIONS],
            "aggregated": aggregated,
            "per_case": results,
            "elapsed_s": round(elapsed_total, 1),
        }
        out_path = Path("experiments/ablation_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemiFA ablation study using OpenAI")
    parser.add_argument("--cases", type=int, default=5,
                        help="Number of test cases to run (1-5, default: 5)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model to use (default: gpt-4o). "
                             "Use gpt-4o-mini for cheaper runs.")
    parser.add_argument("--save-results", action="store_true", default=True)
    args = parser.parse_args()
    main(n_cases=args.cases, save_results=args.save_results,
         api_key=args.api_key, model=args.model)
