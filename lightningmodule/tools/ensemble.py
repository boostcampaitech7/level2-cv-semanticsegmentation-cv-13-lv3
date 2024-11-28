import torch
import numpy as np
import inspect

def uniform_soup(model, paths, device="cpu", by_name=False):
    if not isinstance(paths, list):
        paths = [paths]

    model = model.to(device)
    model_dict = model.state_dict()
    soups = {key: [] for key in model_dict}

    for path in paths:
        checkpoint = torch.load(path, map_location=device)
        weights = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
        if by_name:
            weights = {k: v for k, v in weights.items() if k in model_dict}
        for k, v in weights.items():
            soups[k].append(v)

    for k, v in soups.items():
        if len(v) > 0:
            soups[k] = torch.stack(v).mean(dim=0).to(v[0].dtype)
    
    model_dict.update(soups)
    model.load_state_dict(model_dict)
    return model

def greedy_soup_weights(paths, device="cpu"):
    combined_weights = None
    for path in paths:
        print(f"Loading checkpoint: {path}")
        try:
            checkpoint = torch.load(path, map_location=device)
            if "state_dict" in checkpoint:
                weights = checkpoint["state_dict"]
            else:
                weights = checkpoint
        except ModuleNotFoundError:
            print(f"ModuleNotFoundError: Ignoring class definition issues in {path}.")
            checkpoint = torch.load(path, map_location=device)
            weights = checkpoint.get("state_dict", checkpoint)
        
        if combined_weights is None:
            combined_weights = {k: v.clone() for k, v in weights.items()}
        else:
            for key in combined_weights:
                combined_weights[key] += weights[key]

    for key in combined_weights:
        combined_weights[key] /= len(paths)

    return combined_weights

def greedy_soup(model, paths, data, metric, device="cpu", update_greedy=False, compare=np.greater_equal, by_name=False):
    if not isinstance(paths, list):
        paths = [paths]

    score, soup = None, []
    model = model.to(device)
    model.eval()
    model_dict = model.state_dict()

    for path in paths:
        if update_greedy:
            checkpoint = torch.load(path, map_location=device)
            weights = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
            if by_name:
                weights = {k: v for k, v in weights.items() if k in model_dict}
            if soup:
                weights = {k: (torch.stack([weights[k], soup[k]]).mean(dim=0).to(weights[k].dtype)) for k in model_dict}
            model_dict.update(weights)
            model.load_state_dict(model_dict)
        else:
            model = uniform_soup(model, soup + [path], device=device, by_name=by_name)

        # Evaluate model
        history = []
        for x, y in data:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                preds = model(x)
                score_val = metric(y, preds)
                history.append(score_val)

        avg_score = np.mean(history)
        if score is None or compare(avg_score, score):
            score = avg_score
            if update_greedy:
                soup = weights
            else:
                soup.append(path)

    if update_greedy:
        model_dict.update(soup)
        model.load_state_dict(model_dict)
    else:
        model = uniform_soup(model, soup, device=device, by_name=by_name)

    print(f"Greedy Soup Best Score: {score:.4f}")
    return model