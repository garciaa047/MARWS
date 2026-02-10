# MARWS - Multi-Agent Robotic Warehouse System

Franka Emika Panda robot arm trained with PPO (Stable Baselines3) to pick and place packages.

## Setup

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Train

```bash
python -m training.train_single_agent --timesteps 5000000
```

Resume from a saved model:

```bash
python -m training.train_single_agent --resume models/staged/latest_model.zip
```

Monitor with TensorBoard:

```bash
tensorboard --logdir models/staged/logs
```

## Evaluate

```bash
python -m scripts.evaluate --model models/staged/best_model.zip --episodes 10
```

Without rendering:

```bash
python -m scripts.evaluate --model models/staged/best_model.zip --episodes 10 --no-render
```

## View Scene

```bash
python -m scripts.view_scene
```

## Tests

```bash
pytest tests/test_env.py -v
```
