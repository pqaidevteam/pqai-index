"""Server
Attributes:
    app (fastapi.applications.FastAPI): FastAPI instance
    PORT (int): Port number
"""

import os
import json
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

INDEXES_FOLDER = os.environ["INDEXES_DIR"]
assert os.path.isdir(INDEXES_FOLDER)
INDEX_DIR = IndexStorage(INDEXES_FOLDER)
INDEXES = reduce(add, [INDEX_DIR.get(name) for name in INDEX_DIR.available()])

app = FastAPI()


@app.get("/search")

# pylint: disable=C0103
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
