"""
AISPM - Script 03: QLoRA Fine-Tuning für ASA-Klassifikation
============================================================
Trainiert Modelle D (Llama-3.1-8B) und E (Qwen2.5-7B) auf ASA-Task.
Ground Truth: asa_klasse direkt aus DB — keine manuelle Annotation nötig.

Usage:
  python3 03_finetune.py --model D
  python3 03_finetune.py --model E
  python3 03_finetune.py --model D --epochs 3 --batch_size 4
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

DATA_DIR    = Path("PATH/TO/DATA")
MODEL_CACHE = Path("PATH/TO/MODEL_CACHE")

MODELL_CONFIGS = {
    "D": {
        "name":       "llama-3.1-8b",
        "hf_id":      "meta-llama/Llama-3.1-8B-Instruct",
        "output_dir": str(DATA_DIR / "lora_llama"),
    },
    "E": {
        "name":       "qwen2.5-7b",
        "hf_id":      "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": str(DATA_DIR / "lora_qwen"),
    },
}

SYSTEM_PROMPT = """Du bist ein erfahrener Anästhesist.
Bestimme die ASA-Klassifikation (1-5) des Patienten aus der klinischen Dokumentation.
Antworte NUR mit einem JSON-Objekt: {"asa": 1-5, "reasoning": "Begründung"}"""


def erstelle_trainingsbeispiel(row: pd.Series, tokenizer) -> dict:
    """Konvertiert eine Zeile zu einem Chat-Training-Beispiel."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": str(row["input_text"])},
        {"role": "assistant", "content": json.dumps(
            {"asa": int(row["gt_asa"]),
             "reasoning": "Aus klinischer Dokumentation extrahiert."},
            ensure_ascii=False
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         choices=["D", "E"], required=True)
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=2)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--max_samples",   type=int,   default=None)
    args = parser.parse_args()

    cfg = MODELL_CONFIGS[args.model]
    print(f"\n{'='*60}")
    print(f"QLoRA Fine-Tuning: {cfg['name']} | ASA-Klassifikation")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.learning_rate}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"{'='*60}")

    # Unsloth prüfen
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        print("✓ Unsloth verfügbar")
    except ImportError:
        print("ERROR: pip install unsloth --break-system-packages")
        return

    from trl import SFTTrainer, TrainingArguments

    # Modell laden
    print(f"\nLade: {cfg['hf_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["hf_id"],
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        cache_dir=str(MODEL_CACHE),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainierbar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gesamt      = sum(p.numel() for p in model.parameters())
    print(f"  Trainierbare Parameter: {trainierbar:,} ({trainierbar/gesamt*100:.1f}%)")

    # Daten laden
    print("\nLade Training-Daten...")
    df = pd.read_csv(DATA_DIR / "asa_dataset.csv")
    train_df = df[df["split"] == "train"].copy()
    train_df = train_df[train_df["input_text"].str.len() > 30]

    if args.max_samples:
        train_df = train_df.sample(min(args.max_samples, len(train_df)), random_state=42)

    print(f"  Training-Fälle: {len(train_df)}")
    print(f"  ASA-Verteilung (Training):")
    for k, n in train_df["gt_asa"].value_counts().sort_index().items():
        print(f"    ASA {k}: {n} ({n/len(train_df)*100:.1f}%)")

    # Dataset erstellen
    beispiele = [erstelle_trainingsbeispiel(row, tokenizer) for _, row in train_df.iterrows()]
    hf_dataset = Dataset.from_list(beispiele)

    # Beispiel-Token-Länge
    sample_len = len(tokenizer(beispiele[0]["text"])["input_ids"])
    print(f"  Beispiel-Token-Länge: {sample_len}")

    # Output-Verzeichnis
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=50,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=25,
            save_steps=200,
            save_total_limit=2,
            output_dir=cfg["output_dir"],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        ),
    )

    print(f"\nStarte Training...")
    print(f"  Effektiver Batch: {args.batch_size * args.grad_accum}")
    stats = trainer.train()

    print(f"\n✓ Training abgeschlossen!")
    print(f"  Dauer:      {stats.metrics['train_runtime']/60:.1f} min")
    print(f"  Final Loss: {stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  VRAM:       {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Speichern
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # Konfig-Datei für Reproduzierbarkeit
    konfig = {
        "modell":           args.model,
        "hf_id":            cfg["hf_id"],
        "task":             "ASA-Klassifikation",
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "learning_rate":    args.learning_rate,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "n_training":       len(train_df),
        "final_loss":       stats.metrics.get("train_loss"),
        "training_min":     stats.metrics["train_runtime"] / 60,
    }
    with open(f"{cfg['output_dir']}/training_config.json", "w") as f:
        json.dump(konfig, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Adapter + Konfig: {cfg['output_dir']}")


if __name__ == "__main__":
    main()