.PHONY: test dev up
test:
	python -m pytest -q
dev:  # CPU test doubles, no auth, no GPU: for local API exploration only
	SPEECH_BACKEND=fake SPEECH_AUTH_DISABLED=true SPEECH_DATA_DIR=./.data uvicorn app.main:app_factory --factory --port 8000
up:
	docker compose --profile tls up -d --build
