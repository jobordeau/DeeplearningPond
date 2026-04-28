import os

import torch


def save_model(model, winrate, folder="models", prefix="model"):
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, f"{prefix}_{winrate:.2f}.pth")
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model, path, map_location=None):
    if map_location is None:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def update_best_models(winrate, model_path, best_models, top_k=5):
    best_models.append((winrate, model_path))
    best_models = sorted(best_models, key=lambda x: x[0], reverse=True)
    keep = best_models[:top_k]
    discard = best_models[top_k:]
    for _, path in discard:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return keep
