.PHONY: requirements
req:
	@uv pip freeze > requirements.txt

.PHONY: run-app
run:
	@streamlit run src/Hello.py
