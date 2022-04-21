train:
	python main.py train

evaluate:
	python main.py evaluate

data:
	python src/data/make_data.py

clean:
	rm -rf __pycache__

requirements: requirements.txt
	pip install -r requirements.txt

env:
	python -m venv venv

