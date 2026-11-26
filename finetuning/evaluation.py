"""Single next-token evaluation on the test split.

A test item counts as correct when the one generated token, stripped of
spaces, equals the definiendum; only these definitions are traced.
"""

import os
import torch


def evaluate(model, tokenizer, items):
    """``(accuracy, correct)``; ``correct`` holds the generated texts."""
    model.eval()
    device = next(model.parameters()).device
    correct = []
    for item in items:
        inputs = tokenizer([item["text"]], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1,
                                    pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = generated.split(item["text"])[-1]
        if prediction.strip() == item["label"].strip():
            correct.append(generated)
    accuracy = len(correct) / len(items) if items else 0.0
    return accuracy, correct


def write_report(config, paths, timestamp, accuracy, correct, template):
    """Settings, accuracy and the correctly predicted test items, as text."""
    model_tag = config.run_name.split("/")[0]
    path = os.path.join(paths["results"], f"correct_output_model_{model_tag}.txt")
    lines = [
        f"Experiment Date: {timestamp}",
        f"Checkpoints path: {paths['checkpoints']}",
        f"Model weights unmerged: {paths['unmerged']}",
        f"Model weights: {paths['merged']}",
        f"Output results: {paths['results']}",
        "Training attributes:",
        f"Method: {config.method}",
        f"Loss: {config.loss}",
        f"Epochs: {config.epochs}",
        f"Batch size: {config.batch_size}",
        f"Save steps: {config.save_steps}",
        f"Save total limit: {config.save_total_limit}",
        f"R: {config.r}",
        f"Lora alpha: {config.lora_alpha}",
        f"Lora dropout: {config.lora_dropout}",
        f"Model: {config.model_name}",
        f"Prompt: {template}",
        f"Accuracy: {accuracy}",
        f"Number Correct predictions: {len(correct)}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for item in correct:
            f.write(item + "\n")
    return path
