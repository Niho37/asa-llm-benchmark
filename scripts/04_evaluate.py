""" AISPM - Script 04: Evaluation & Paper-Tabellen
================================================================
Liest alle Inferenz-Resultate aus results/ und berechnet:

  Primärmetriken:
    - Quadratic Weighted Kappa (primär)
    - MAE
    - ±1 Accuracy
    - Exact Match Accuracy
    - Parse Error Rate

  Fehleranalyse:
    - Confusion Matrix (5×5) pro Modell/Technik/Prompt
    - Per-Class Recall & Precision
    - Signed Error (systematisches Über-/Unterschätzen)

  Output:
    - paper_table.csv         → Haupttabelle fürs Paper
    - per_class_metrics.csv   → Per-Class Recall/Precision
    - confusion_matrices/     → Confusion Matrix pro Run als PNG
    - summary_report.txt      → Lesbare Zusammenfassung

Usage:
  python3 04_evaluate.py                          # alle CSVs in results/
  python3 04_evaluate.py --split test             # nur Test-Set
  python3 04_evaluate.py --prompt P2              # nur Prompt P2
  python3 04_evaluate.py --model mistral-small-24b
  python3 04_evaluate.py --split test --prompt P2 # Test + P2 (Paper-Zahlen)
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # kein Display nötig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

warnings.filterwarnings("ignore")

# ── Pfade ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("PATH/TO/DATA")
RESULTS_DIR = DATA_DIR / "results"
EVAL_DIR    = DATA_DIR / "evaluation"
CM_DIR      = EVAL_DIR / "confusion_matrices"

EVAL_DIR.mkdir(exist_ok=True)
CM_DIR.mkdir(exist_ok=True)

# Modell-Reihenfolge für Paper-Tabelle (A→F)
MODELL_REIHENFOLGE = [
    "phi-4-reasoning-plus",
    "mistral-small-24b",
    "qwen2.5-14b",
    "meditron3-7b",
    "llama-3.1-8b-lora",
    "qwen2.5-7b-lora",
]

# Schöne Modellnamen fürs Paper
MODELL_NAMEN = {
    "phi-4-reasoning-plus":  "Phi-4 Reasoning Plus",
    "mistral-small-24b":     "Mistral Small 24B",
    "qwen2.5-14b":           "Qwen2.5-14B",
    "meditron3-7b":          "Meditron3-7B",
    "llama-3.1-8b-lora":     "Llama-3.1-8B + LoRA",
    "qwen2.5-7b-lora":       "Qwen2.5-7B + LoRA",
}

ASA_KLASSEN = [1, 2, 3, 4, 5]


# ═══════════════════════════════════════════════════════════════════════════════
# Resultate laden
# ═══════════════════════════════════════════════════════════════════════════════

def lade_alle_resultate(
    split_filter:   str = None,
    prompt_filter:  str = None,
    modell_filter:  str = None,
) -> pd.DataFrame:
    """Lädt alle results_*.csv Dateien und kombiniert sie.
    
    Ignoriert Checkpoint-Dateien (ckpt_*).
    Filtert optional nach split, prompt_id, modell.
    """
    csv_files = sorted(RESULTS_DIR.glob("results_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Keine results_*.csv Dateien in {RESULTS_DIR}\n"
            f"Zuerst 02_inference.py laufen lassen."
        )

    print(f"Gefundene CSV-Dateien: {len(csv_files)}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Pflicht-Spalten prüfen
            if "gt_asa" not in df.columns or "pred_asa" not in df.columns:
                print(f"  [SKIP] {f.name} — fehlende Spalten")
                continue
            # Dateiname-basierte Felder ergänzen falls Spalten fehlen
            if "prompt_id" not in df.columns:
                # Versuche aus Dateinamen zu extrahieren (z.B. _P2_)
                for pid in ["P0", "P1", "P2", "P3"]:
                    if f"_{pid}_" in f.name:
                        df["prompt_id"] = pid
                        break
                else:
                    df["prompt_id"] = "unknown"
            if "split" not in df.columns:
                for s in ["test", "val", "train"]:
                    if f"_{s}_" in f.name:
                        df["split"] = s
                        break
                else:
                    df["split"] = "unknown"
            dfs.append(df)
        except Exception as e:
            print(f"  [FEHLER] {f.name}: {e}")

    if not dfs:
        raise ValueError("Keine gültigen CSV-Dateien geladen.")

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Gesamt: {len(all_df)} Zeilen aus {len(dfs)} Dateien\n")

    # Filter anwenden
    if split_filter:
        all_df = all_df[all_df["split"] == split_filter]
        print(f"Filter split={split_filter}: {len(all_df)} Zeilen")
    if prompt_filter:
        all_df = all_df[all_df["prompt_id"] == prompt_filter]
        print(f"Filter prompt={prompt_filter}: {len(all_df)} Zeilen")
    if modell_filter:
        all_df = all_df[all_df["modell"].str.contains(modell_filter, case=False)]
        print(f"Filter modell={modell_filter}: {len(all_df)} Zeilen")

    return all_df


# ═══════════════════════════════════════════════════════════════════════════════
# Metriken berechnen
# ═══════════════════════════════════════════════════════════════════════════════

def berechne_metriken(df: pd.DataFrame) -> dict:
    """Berechnet alle Metriken für einen DataFrame (ein Modell/Technik/Prompt)."""
    n_total = len(df)
    parse_err_rate = df["parse_error"].mean() if "parse_error" in df.columns else float("nan")

    valid = df.dropna(subset=["pred_asa", "gt_asa"]).copy()
    valid["pred_asa"] = valid["pred_asa"].astype(int)
    valid["gt_asa"]   = valid["gt_asa"].astype(int)

    # Nur gültige ASA-Werte (1–5)
    valid = valid[valid["pred_asa"].isin(ASA_KLASSEN) & valid["gt_asa"].isin(ASA_KLASSEN)]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": 0,
            "kappa": float("nan"), "mae": float("nan"),
            "off1": float("nan"), "exact": float("nan"),
            "parse_err": parse_err_rate,
            "signed_error": float("nan"),
        }

    kappa = cohen_kappa_score(valid["gt_asa"], valid["pred_asa"], weights="quadratic")
    mae   = (valid["pred_asa"] - valid["gt_asa"]).abs().mean()
    off1  = (valid["pred_asa"] - valid["gt_asa"]).abs().le(1).mean()
    exact = (valid["pred_asa"] == valid["gt_asa"]).mean()
    signed_error = (valid["pred_asa"] - valid["gt_asa"]).mean()  # + = overcall

    # 95% CI für Kappa via Bootstrap
    kappa_ci = _bootstrap_kappa_ci(valid["gt_asa"].values, valid["pred_asa"].values)

    return {
        "n_total":      n_total,
        "n_valid":      n_valid,
        "kappa":        round(kappa, 4),
        "kappa_ci_low": round(kappa_ci[0], 4),
        "kappa_ci_high": round(kappa_ci[1], 4),
        "mae":          round(mae, 4),
        "off1":         round(off1, 4),
        "exact":        round(exact, 4),
        "parse_err":    round(parse_err_rate, 4),
        "signed_error": round(signed_error, 4),
    }


def _bootstrap_kappa_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05) -> tuple:
    """95% CI für Weighted Kappa via Bootstrap (n=1000)."""
    rng    = np.random.default_rng(42)
    n      = len(y_true)
    kappas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            k = cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic")
            kappas.append(k)
        except Exception:
            pass
    if not kappas:
        return (float("nan"), float("nan"))
    kappas = np.array(kappas)
    return (
        float(np.percentile(kappas, 100 * alpha / 2)),
        float(np.percentile(kappas, 100 * (1 - alpha / 2))),
    )


def berechne_per_class_metriken(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Recall, Precision, F1 pro ASA-Klasse."""
    valid = df.dropna(subset=["pred_asa", "gt_asa"]).copy()
    valid["pred_asa"] = valid["pred_asa"].astype(int)
    valid["gt_asa"]   = valid["gt_asa"].astype(int)

    rows = []
    for asa in ASA_KLASSEN:
        tp = ((valid["gt_asa"] == asa) & (valid["pred_asa"] == asa)).sum()
        fp = ((valid["gt_asa"] != asa) & (valid["pred_asa"] == asa)).sum()
        fn = ((valid["gt_asa"] == asa) & (valid["pred_asa"] != asa)).sum()

        n_gt   = (valid["gt_asa"] == asa).sum()
        n_pred = (valid["pred_asa"] == asa).sum()

        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))

        rows.append({
            "asa_class":  asa,
            "n_gt":       n_gt,
            "n_pred":     n_pred,
            "recall":     round(recall, 4),
            "precision":  round(precision, 4),
            "f1":         round(f1, 4),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix plotten
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    df:        pd.DataFrame,
    title:     str,
    save_path: Path,
):
    """Plottet und speichert eine normalisierte + absolute Confusion Matrix."""
    valid = df.dropna(subset=["pred_asa", "gt_asa"]).copy()
    valid["pred_asa"] = valid["pred_asa"].astype(int)
    valid["gt_asa"]   = valid["gt_asa"].astype(int)

    if len(valid) == 0:
        return

    cm = confusion_matrix(valid["gt_asa"], valid["pred_asa"], labels=ASA_KLASSEN)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, normalize, subtitle in zip(
        axes,
        [False, True],
        ["Absolute Counts", "Row-Normalized (Recall per Class)"]
    ):
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            data = np.where(row_sums > 0, cm / row_sums, 0)
            fmt  = ".2f"
            vmax = 1.0
        else:
            data = cm
            fmt  = "d"
            vmax = cm.max()

        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(ASA_KLASSEN)))
        ax.set_yticks(range(len(ASA_KLASSEN)))
        ax.set_xticklabels([f"ASA {k}" for k in ASA_KLASSEN])
        ax.set_yticklabels([f"ASA {k}" for k in ASA_KLASSEN])
        ax.set_xlabel("Predicted ASA", fontsize=11)
        ax.set_ylabel("True ASA", fontsize=11)
        ax.set_title(subtitle, fontsize=10)

        for i in range(len(ASA_KLASSEN)):
            for j in range(len(ASA_KLASSEN)):
                val = data[i, j]
                text = f"{val:{fmt}}" if not normalize else f"{val:.2f}"
                color = "white" if (normalize and val > 0.5) or (not normalize and val > vmax * 0.6) else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Hauptauswertung
# ═══════════════════════════════════════════════════════════════════════════════

def erstelle_paper_tabelle(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt die Haupttabelle fürs Paper: eine Zeile pro Modell×Technik×Prompt."""
    group_cols = ["modell", "technik", "prompt_id"]
    # Nur Spalten die existieren
    group_cols = [c for c in group_cols if c in df.columns]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        metriken = berechne_metriken(grp)
        row = {**key_dict, **metriken}

        # Schöner Modellname
        modell_raw = key_dict.get("modell", "")
        row["modell_paper"] = MODELL_NAMEN.get(modell_raw, modell_raw)

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sortierung: Modell-Reihenfolge A→F, dann Technik, dann Prompt
    result["sort_key"] = result["modell"].apply(
        lambda m: MODELL_REIHENFOLGE.index(m) if m in MODELL_REIHENFOLGE else 99
    )
    result = result.sort_values(["sort_key", "technik", "prompt_id"]).drop("sort_key", axis=1)
    result = result.reset_index(drop=True)

    return result


def erstelle_per_class_tabelle(df: pd.DataFrame) -> pd.DataFrame:
    """Per-Class Recall/Precision für alle Modell×Technik×Prompt Kombinationen."""
    group_cols = ["modell", "technik", "prompt_id"]
    group_cols = [c for c in group_cols if c in df.columns]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        pc = berechne_per_class_metriken(grp)
        for _, r in pc.iterrows():
            rows.append({**key_dict, **r.to_dict()})

    return pd.DataFrame(rows)


def erstelle_confusion_matrices(df: pd.DataFrame):
    """Plottet Confusion Matrices für alle Kombinationen."""
    group_cols = ["modell", "technik", "prompt_id"]
    group_cols = [c for c in group_cols if c in df.columns]

    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        modell  = key_dict.get("modell", "unknown")
        technik = key_dict.get("technik", "unknown")
        prompt  = key_dict.get("prompt_id", "unknown")
        split   = grp["split"].iloc[0] if "split" in grp.columns else "unknown"

        titel     = f"{MODELL_NAMEN.get(modell, modell)} | {technik} | {prompt} | {split}"
        dateiname = f"cm_{modell}_{technik}_{prompt}_{split}.png"

        plot_confusion_matrix(grp, titel, CM_DIR / dateiname)
        print(f"  Confusion Matrix gespeichert: {dateiname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════════════

def erstelle_summary_report(paper_table: pd.DataFrame, per_class: pd.DataFrame) -> str:
    """Erstellt einen lesbaren Text-Report."""
    lines = []
    lines.append("=" * 70)
    lines.append("AISPM ASA-KLASSIFIKATION — EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Haupttabelle
    lines.append("HAUPTTABELLE (Paper-Metriken)")
    lines.append("-" * 70)

    display_cols = ["modell_paper", "technik", "prompt_id",
                    "n_valid", "kappa", "kappa_ci_low", "kappa_ci_high",
                    "exact", "off1", "mae", "parse_err", "signed_error"]
    display_cols = [c for c in display_cols if c in paper_table.columns]

    # Formatierte Ausgabe
    for _, row in paper_table.iterrows():
        lines.append(
            f"  {row.get('modell_paper', ''):<28} "
            f"{row.get('technik', ''):<12} "
            f"{row.get('prompt_id', ''):<4} | "
            f"κ={row.get('kappa', float('nan')):.3f} "
            f"[{row.get('kappa_ci_low', float('nan')):.3f}–{row.get('kappa_ci_high', float('nan')):.3f}] | "
            f"Exact={row.get('exact', float('nan')):.3f} | "
            f"±1={row.get('off1', float('nan')):.3f} | "
            f"MAE={row.get('mae', float('nan')):.3f} | "
            f"Err={row.get('signed_error', float('nan')):+.3f}"
        )
    lines.append("")

    # Best Model
    valid_rows = paper_table.dropna(subset=["kappa"])
    if len(valid_rows) > 0:
        best = valid_rows.loc[valid_rows["kappa"].idxmax()]
        lines.append("BESTES MODELL (nach Kappa):")
        lines.append(
            f"  {best.get('modell_paper', '')} | {best.get('technik', '')} | "
            f"{best.get('prompt_id', '')} → κ={best['kappa']:.3f}"
        )
        lines.append("")

    # Per-Class Recall (für bestes Modell)
    if len(valid_rows) > 0 and len(per_class) > 0:
        best_modell  = best.get("modell", "")
        best_technik = best.get("technik", "")
        best_prompt  = best.get("prompt_id", "")

        mask = (
            (per_class.get("modell", pd.Series()) == best_modell) &
            (per_class.get("technik", pd.Series()) == best_technik) &
            (per_class.get("prompt_id", pd.Series()) == best_prompt)
        )
        pc_best = per_class[mask]

        if len(pc_best) > 0:
            lines.append(f"PER-CLASS METRIKEN (bestes Modell: {best.get('modell_paper', '')})")
            lines.append("-" * 50)
            lines.append(f"  {'ASA':<6} {'n_GT':<7} {'n_Pred':<8} {'Recall':<9} {'Prec':<9} {'F1'}")
            for _, r in pc_best.iterrows():
                lines.append(
                    f"  ASA {int(r['asa_class']):<3} "
                    f"{int(r['n_gt']):<7} "
                    f"{int(r['n_pred']):<8} "
                    f"{r['recall']:.3f}    "
                    f"{r['precision']:.3f}    "
                    f"{r['f1']:.3f}"
                )
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt-Ablation Tabelle (P0 vs P2 Vergleich)
# ═══════════════════════════════════════════════════════════════════════════════

def erstelle_ablation_tabelle(df: pd.DataFrame) -> pd.DataFrame:
    """Vergleicht P0 vs P2 für dasselbe Modell/Technik."""
    prompts = df["prompt_id"].unique() if "prompt_id" in df.columns else []
    if len(prompts) < 2:
        return pd.DataFrame()

    rows = []
    group_cols = ["modell", "technik"]
    group_cols = [c for c in group_cols if c in df.columns]

    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        for pid in ["P0", "P1", "P2", "P3"]:
            sub = grp[grp["prompt_id"] == pid] if "prompt_id" in grp.columns else pd.DataFrame()
            if len(sub) == 0:
                continue
            m = berechne_metriken(sub)
            rows.append({
                **key_dict,
                "prompt_id": pid,
                "kappa":     m["kappa"],
                "mae":       m["mae"],
                "exact":     m["exact"],
                "n_valid":   m["n_valid"],
            })

    if not rows:
        return pd.DataFrame()

    abl = pd.DataFrame(rows)

    # Delta P2 vs P0
    delta_rows = []
    for keys, grp in abl.groupby(group_cols):
        p0 = grp[grp["prompt_id"] == "P0"]
        p2 = grp[grp["prompt_id"] == "P2"]
        if len(p0) > 0 and len(p2) > 0:
            delta_rows.append({
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "kappa_P0":    p0["kappa"].values[0],
                "kappa_P2":    p2["kappa"].values[0],
                "delta_kappa": round(p2["kappa"].values[0] - p0["kappa"].values[0], 4),
                "mae_P0":      p0["mae"].values[0],
                "mae_P2":      p2["mae"].values[0],
                "delta_mae":   round(p2["mae"].values[0] - p0["mae"].values[0], 4),
            })

    return pd.DataFrame(delta_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AISPM ASA Evaluation")
    parser.add_argument("--split",   default=None, help="Nur diesen Split auswerten (train/val/test)")
    parser.add_argument("--prompt",  default=None, help="Nur diesen Prompt auswerten (P0/P1/P2/P3)")
    parser.add_argument("--model",   default=None, help="Nur dieses Modell (substring)")
    parser.add_argument("--no_plots", action="store_true", help="Keine Confusion Matrices plotten")
    args, _ = parser.parse_known_args()

    print("\n" + "=" * 70)
    print("AISPM Evaluation Script 04")
    print("=" * 70 + "\n")

    # ── Daten laden ───────────────────────────────────────────────────────────
    df = lade_alle_resultate(
        split_filter=args.split,
        prompt_filter=args.prompt,
        modell_filter=args.model,
    )

    if len(df) == 0:
        print("Keine Daten nach Filterung — prüfe --split / --prompt / --model")
        return

    print(f"\nVerfügbare Splits:    {sorted(df['split'].unique()) if 'split' in df.columns else 'N/A'}")
    print(f"Verfügbare Modelle:   {sorted(df['modell'].unique()) if 'modell' in df.columns else 'N/A'}")
    print(f"Verfügbare Prompts:   {sorted(df['prompt_id'].unique()) if 'prompt_id' in df.columns else 'N/A'}")
    print(f"Verfügbare Techniken: {sorted(df['technik'].unique()) if 'technik' in df.columns else 'N/A'}")
    print()

    # ── Paper-Tabelle ─────────────────────────────────────────────────────────
    print("Berechne Paper-Tabelle...")
    paper_table = erstelle_paper_tabelle(df)
    out_path = EVAL_DIR / "paper_table.csv"
    paper_table.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    # ── Per-Class Metriken ────────────────────────────────────────────────────
    print("Berechne Per-Class Metriken...")
    per_class = erstelle_per_class_tabelle(df)
    out_path = EVAL_DIR / "per_class_metrics.csv"
    per_class.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    # ── Ablation Tabelle ──────────────────────────────────────────────────────
    if "prompt_id" in df.columns and df["prompt_id"].nunique() > 1:
        print("Berechne Prompt-Ablation (P0 vs P2)...")
        ablation = erstelle_ablation_tabelle(df)
        if len(ablation) > 0:
            out_path = EVAL_DIR / "prompt_ablation.csv"
            ablation.to_csv(out_path, index=False)
            print(f"  → {out_path}")

    # ── Confusion Matrices ────────────────────────────────────────────────────
    if not args.no_plots:
        print("Erstelle Confusion Matrices...")
        erstelle_confusion_matrices(df)

    # ── Summary Report ────────────────────────────────────────────────────────
    print("Erstelle Summary Report...")
    report = erstelle_summary_report(paper_table, per_class)
    report_path = EVAL_DIR / "summary_report.txt"
    report_path.write_text(report)
    print(f"  → {report_path}")

    # ── Report ausgeben ───────────────────────────────────────────────────────
    print("\n" + report)

    print(f"\nAlle Outputs in: {EVAL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()