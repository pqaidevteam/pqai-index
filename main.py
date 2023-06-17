"""Server
Attributes:
    app (fastapi.applications.FastAPI): FastAPI instance
    PORT (int): Port number
"""

import os
import json
from pathlib import Path
import dotenv
from operator import add
from functools import reduce
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

dotenv.load_dotenv()

from core.storage import IndexStorage

LOGGING_FORMAT = "%(levelprefix)s %(client_addr)s %(status_code)s"
LOGGING_CONFIG["formatters"]["access"]["fmt"] = LOGGING_FORMAT

APP_DIR = Path(__file__).parent
INDEXES_FOLDER = (APP_DIR / "indexes").resolve()
assert os.path.isdir(INDEXES_FOLDER), f"Cannot find indexes directory: {INDEXES_FOLDER}"
index_storage = IndexStorage(INDEXES_FOLDER)
indexes = reduce(add, [index_storage.get(name) for name in index_storage.available()])

app = FastAPI()

@app.get("/search")
async def search(mode: str, query: str, n: Optional[int] = 10, index: Optional[str] = None):
    """Converts the query into vector and returns top n similar indexes"""

    if mode == "vector":
        try:
            qvec = json.loads(query)
            if index is None:
                target_indexes = indexes
            else:
                target_indexes = index_storage.get(index)
            results = reduce(add, [idx.search(qvec, n) for idx in target_indexes])
            results.sort(key=lambda r: r[1])
            return {"query": qvec, "results": results[:n]}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON query")
    else:
        raise HTTPException(status_code=400, detail="Invalid search mode")


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
