"""Server
Attributes:
    app (fastapi.applications.FastAPI): FastAPI instance
    PORT (int): Port number
"""

import os
import json
from pathlib import Path
import dotenv
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI, HTTPException

dotenv.load_dotenv()

# pylint: disable=C0411

from operator import add
from functools import reduce

LOGGING_FORMAT = "%(levelprefix)s %(client_addr)s %(status_code)s"
LOGGING_CONFIG["formatters"]["access"]["fmt"] = LOGGING_FORMAT

from core.storage import IndexStorage

APP_DIR = Path(__file__).parent
INDEXES_FOLDER = (APP_DIR / "indexes").resolve()
assert os.path.isdir(INDEXES_FOLDER), f"Cannot find indexes directory: {INDEXES_FOLDER}"
INDEX_DIR = IndexStorage(INDEXES_FOLDER)
INDEXES = reduce(add, [INDEX_DIR.get(name) for name in INDEX_DIR.available()])

app = FastAPI()


@app.get("/search")
async def search(mode: str, query: str, n: int):
    """Converts the query into vector and returns top n similar indexes"""

    if mode == "vector":
        try:
            qvec = json.loads(query)
            results = reduce(add, [idx.search(qvec, n) for idx in INDEXES])
            results.sort(key=lambda r: r[1])
            return {"query": qvec, "results": results[:n]}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON query")
    else:
        raise HTTPException(status_code=400, detail="Invalid search mode")


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
