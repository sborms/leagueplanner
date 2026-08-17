.PHONY: example

push:
	git add .
	git commit -m "$(m)"
	git push

brush:
	uv run ruff check --select I --fix .
	uv run ruff format .
	uv run ruff check .

install:
	pip install uv
	uv venv
	uv sync

freeze:
	uv sync
	uv pip compile pyproject.toml -o requirements.txt >/dev/null

web:
	uv run streamlit run app.py

example:
	rm -rf example/output
	uv run 2rr --input-file "example/input.xlsx" --output-folder "example/output" --seed 321 --n-iterations 5000 --games-per-opponent 2

time:
	uv run timings.py

profile:
	python -m cProfile -o profile.out timings.py
	snakeviz profile.out
	
experiment:
	uv run 2rr \
		--input-file "experiments/real_life_test_input_3_matchups_stripped.xlsx" \
		--output-folder "experiments/difficile" \
		--seed 505 \
		--n-iterations 10000 \
		--r-max 4 \
		--m 2 \
		--games-per-opponent 3
