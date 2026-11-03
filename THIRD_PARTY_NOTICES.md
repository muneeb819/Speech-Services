# Third-party components and model rights

This starter contains reference scripts only. It does not bundle model weights, voice recordings, PyTorch, CUDA libraries or other third-party packages.

When installing or redistributing a complete deployment, collect the actual license texts and attribution notices for every included component.

| Component | Source | Checked position / action |
|---|---|---|
| faster-whisper | https://github.com/SYSTRAN/faster-whisper | MIT runtime; review CTranslate2 and other dependencies. |
| Whisper large-v3-turbo | https://huggingface.co/openai/whisper-large-v3-turbo | MIT at checked revision. |
| Kokoro inference | https://github.com/hexgrad/kokoro | Apache-2.0; review bundled dependencies separately. |
| Kokoro weights | https://huggingface.co/hexgrad/Kokoro-82M | Apache-2.0 at checked revision; preserve required notices and attribution. |
| Misaki | https://github.com/hexgrad/misaki | Apache-2.0 project; check lexicon/data and optional dependency terms. |
| eSpeak NG | https://github.com/espeak-ng/espeak-ng | GPL-family system dependency; review distribution/linking obligations. |
| Chatterbox | https://github.com/resemble-ai/chatterbox | MIT code. |
| Chatterbox Turbo | https://huggingface.co/ResembleAI/chatterbox-turbo | MIT at checked revision; upstream generation applies a Perth watermark. |
| Perth and other Chatterbox dependencies | Refer to the exact Chatterbox dependency lock/revision | Review individually before redistribution. |

An MIT/Apache model license is not permission to impersonate a person or use a voice without required consent. CC BY and Apache attribution obligations must be met. Non-commercial model licenses cannot be ignored because the runtime is permissive.

Record the exact downloaded revisions and keep a software bill of materials for production. This file is an engineering checklist, not legal advice or a complete license bundle.
