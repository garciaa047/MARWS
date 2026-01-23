TEmp
Try running the prog
C:\MARWS\venv\Scripts\python.exe training/train_single_agent.py --iterations 5000 --resume

View scene
C:\MARWS\venv\Scripts\python.exe scripts\view_scene.py

Check the checkpoint
C:\MARWS\venv\Scripts\python.exe scripts/evaluate.py models/single_agent --episodes 10


I had to add os.path.abspath("models/single_agent") to 

    parser.add_argument("--checkpoint-dir", type=str, default=os.path.abspath("models/single_agent"))

in train_single_agent line 58