"""Expose fitted categories and uploaded-image batch inspection through FastAPI."""

import asyncio
import io
import json
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .gate import RefCalGate
from .routing import select_cloud


def create_app(gate_path, config):
    """Return a FastAPI app using an NPZ gate and cloud/routing configuration.

    GET /categories lists fitted categories. POST /inspect accepts multipart
    files, a JSON categories string in matching order, and optional cloud_fraction.
    It returns cloud_calls, count, and ordered predictions with edge/cloud sources.
    """
    from PIL import Image

    gate = RefCalGate.load(gate_path, config.encoder.device)
    inspector = None
    lock = threading.Lock()
    app = FastAPI(title="RefCal-Gate")

    def inspect_batch(payloads, categories, fraction):
        nonlocal inspector
        # Serialize shared model access while the async endpoint stays responsive.
        with lock:
            images = []
            for raw in payloads:
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(image.convert("RGB"))
            rows = [gate.predict_image(image, category) for image, category in zip(images, categories)]
            selected = select_cloud(rows, fraction)
            if selected and inspector is None:
                from .cloud import CloudInspector

                inspector = CloudInspector(config.cloud)
            for index, (image, row) in enumerate(zip(images, rows)):
                row["index"] = index
                row["edge_predicted"] = row["predicted"]
                row["source"] = "cloud" if index in selected else "edge"
                if index in selected:
                    result = inspector.predict(image)
                    row["predicted"] = result["predicted"]
                    row["cloud"] = result
            return {"cloud_calls": len(selected), "count": len(rows), "predictions": rows}

    @app.get("/categories")
    def categories():
        """Return the category names available in the fitted reference memory."""
        return {"categories": sorted(gate.memory.references)}

    @app.post("/inspect")
    async def inspect(files: list[UploadFile] = File(...), categories: str = Form(...),
                      cloud_fraction: float | None = Form(None)):
        """Validate multipart inputs and return a batch result from a worker thread."""
        try:
            names = json.loads(categories)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "Categories must be a JSON array of strings") from exc
        if not isinstance(names, list) or not names or len(names) != len(files):
            raise HTTPException(400, "Provide one category for each uploaded image")
        if any(not isinstance(name, str) or name not in gate.memory.references for name in names):
            raise HTTPException(400, "Every image category must have a fitted reference memory")
        fraction = config.routing.cloud_fraction if cloud_fraction is None else cloud_fraction
        if not 0 <= fraction <= 1:
            raise HTTPException(400, "Cloud fraction must be in [0, 1]")
        payloads = [await upload.read() for upload in files]
        try:
            return await asyncio.to_thread(inspect_batch, payloads, names, fraction)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, "Image inspection failed; check inputs and server configuration") from exc

    return app


def serve(gate_path, config, host, port):
    """Run the inspection app with Uvicorn at the supplied host and port."""
    import uvicorn

    uvicorn.run(create_app(gate_path, config), host=host, port=port)
