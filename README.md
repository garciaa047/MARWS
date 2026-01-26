Temp 

**RUN THE PROGRAM** 

1. python -m venv venv

2. venv\Scripts\activate.bat

3. pip install -r requirements.txt

4. python -m training.train_single_agent --iterations 250

**HOW TO VIEW SCENE**

View scene python -m scripts.view_scene

**EVALUATE**

python -m scripts.evaluate --episodes 10

**OTHER**

I had to add os.path.abspath("models/single_agent") to

parser.add_argument("--checkpoint-dir", type=str, default=os.path.abspath("models/single_agent"))
in train_single_agent line 58