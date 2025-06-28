"""Baseline data and configuration management for schedule optimization."""

# Baselines data
baselines = {
    "3A021JEHN02756": {
        "cifar-dense-vk": {
            "omp": 940,
            "vk": 11.4,
        },
        "cifar-sparse-vk": {
            "omp": 45.8,
            "vk": 44.9,
        },
        "tree-vk": {
            "omp": 14.2,
            "vk": 58.7,
        },
    },
    "9b034f1b": {
        "cifar-dense-vk": {
            "omp": 730,
            "vk": 12.1,
        },
        "cifar-sparse-vk": {
            "omp": 53.2,
            "vk": 27.9,
        },
        "tree-vk": {
            "omp": 12.7,
            "vk": 47.2,
        },
    },
    "jetson": {
        "cifar-dense-cu": {
            "omp": 23.5,
            "cu": 5.48,
        },
        "cifar-sparse-cu": {
            "omp": 486,
            "cu": 27.2,
        },
        "tree-cu": {
            "omp": 16.2,
            "cu": 5.42,
        },
    },
    "jetsonlowpower": {
        "cifar-dense-cu": {
            "omp": 58.5,
            "cu": 23.6,
        },
        "cifar-sparse-cu": {
            "omp": 1042,
            "cu": 101,
        },
        "tree-cu": {
            "omp": 39.7,
            "cu": 7.28,
        },
    },
}


def get_baseline_for_config(device, app, backend):
    """Get baseline times for the given device-app-backend configuration."""
    app_backend_key = f"{app}-{backend}"
    try:
        baseline_data = baselines[device][app_backend_key]
        # Find the fastest baseline (minimum of omp and backend-specific time)
        omp_time = baseline_data["omp"]
        gpu_time = baseline_data[backend]
        fastest_time = min(omp_time, gpu_time)
        return {"omp": omp_time, backend: gpu_time, "fastest": fastest_time}
    except KeyError:
        print(f"Warning: No baseline found for {device}/{app}/{backend}")
        return None


def get_num_stages_for_app(app_name):
    """Get the number of stages for the given application."""
    stage_counts = {
        "tree": 7,
        "cifar-dense": 9,
        "cifar-sparse": 9,
    }

    # Extract the base app name without backend suffix
    base_app = app_name.split("-")[0] if "-" in app_name else app_name

    # For cifar apps, both dense and sparse variants have the same number of stages
    if base_app == "cifar":
        return stage_counts.get("cifar-dense", 9)

    return stage_counts.get(base_app, 9)  # Default to 9 if not found 