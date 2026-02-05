"""Run with: python -m suno_api"""

import uvicorn

uvicorn.run("suno_api.server:app", host="0.0.0.0", port=8100, reload=True)
