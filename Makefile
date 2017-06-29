init:
	pip install -r requirements.txt

test:
	nosetests tests --nologcapture --rednose -d

run:
	python3 src/main.py
help:
	python3 src/main.py --help 
