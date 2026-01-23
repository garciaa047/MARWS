Temp Try running the prog venv\Scripts\activate.bat

pip install -r requirements.txt

python -m training.train_single_agent --iterations 250

View scene python -m scripts.view_scene

Check the checkpoint python -m scripts.evaluate --episodes 10

I had to add os.path.abspath("models/single_agent") to

parser.add_argument("--checkpoint-dir", type=str, default=os.path.abspath("models/single_agent"))
in train_single_agent line 58