"""Keyless live test for VobizFrameSerializer: pre-recorded speech and local Silero VAD.

Answers a real Vobiz call, plays a long greeting, and when the far end talks over
it, sends an interruption and plays a short reply. /events shows what crossed the wire.
"""

import array
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
from xml.sax.saxutils import escape
from urllib.parse import urlencode
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).parent
GREETING = (HERE / "greeting.pcm").read_bytes()
REPLY = (HERE / "reply.pcm").read_bytes()
EVENTS: list[dict] = []
CAPTURE: dict[str, bytearray] = {}
T0 = time.monotonic()
app = FastAPI()


def log(what: str, **extra) -> None:
    EVENTS.append({"t": round(time.monotonic() - T0, 3), "what": what, **extra})
    print(EVENTS[-1], flush=True)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/capture/{name}")
async def capture(name: str) -> Response:
    return Response(content=bytes(CAPTURE.get(name, b"")), media_type="application/octet-stream")


@app.get("/events")
async def events() -> list[dict]:
    return EVENTS


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request) -> Response:
    form = dict(await request.form()) if request.method == "POST" else {}
    q = dict(request.query_params)
    log("answer url requested", event=form.get("Event"), status=form.get("CallStatus"), q=q.get("name"))
    url = f"wss://{request.headers['host']}/ws?{urlencode(q)}"
    ct = q.get("ct", "audio/x-mulaw;rate=8000")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        f'<Stream bidirectional="true" keepCallAlive="true" contentType="{ct}">'
        f"{escape(url)}</Stream></Response>"
    )
    return Response(content=xml, media_type="application/xml")


def _swap(b64: str) -> str:
    a = array.array("h", base64.b64decode(b64))
    a.byteswap()
    return base64.b64encode(a.tobytes()).decode()


class Logged(VobizFrameSerializer):
    played = 0
    name_ = ""
    swap = False

    async def serialize(self, frame):
        out = await super().serialize(frame)
        if out:
            msg = json.loads(out)
            event = msg.get("event")
            if event == "playAudio":
                self.played += 1
                if self.played == 1:
                    log("first playAudio sent", fmt=msg["media"]["contentType"], rate=msg["media"]["sampleRate"], bytes=len(base64.b64decode(msg["media"]["payload"])))
                if self.swap and msg["media"]["contentType"] == "audio/x-l16":
                    msg["media"]["payload"] = _swap(msg["media"]["payload"])
                    out = json.dumps(msg)
            else:
                log(f"sent {event}", play_audio_so_far=self.played)
        return out

    async def deserialize(self, data):
        msg = json.loads(data)
        if msg.get("event") != "media":
            log(f"received {msg.get('event')}", raw=str(msg)[:160])
        elif self.swap and msg.get("media", {}).get("payload"):
            msg["media"]["payload"] = _swap(msg["media"]["payload"])
            data = json.dumps(msg)
        frame = await super().deserialize(data)
        if frame is not None and msg.get("event") == "media":
            CAPTURE.setdefault(self.name_, bytearray()).extend(frame.audio)
        if frame is not None and msg.get("event") == "dtmf":
            log("InputDTMFFrame", digit=frame.button.value)
        return frame


class Script(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._interrupted = False
        self._replied = False

    async def play(self, clip: bytes) -> None:
        for i in range(0, len(clip), 960):
            await self.push_frame(
                OutputAudioRawFrame(audio=clip[i : i + 960], sample_rate=24000, num_channels=1)
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame) and not self._interrupted:
            self._interrupted = True
            log("VAD: far end started talking; interrupting")
            await self.broadcast_interruption()
        elif (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and self._interrupted
            and not self._replied
        ):
            self._replied = True
            log("VAD: far end stopped talking; playing reply")
            await self.play(REPLY)
        await self.push_frame(frame, direction)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    start = json.loads(await websocket.receive_text())
    log("start event", start=start.get("start"))
    q = dict(websocket.query_params)
    fmt = start["start"]["mediaFormat"]
    in_rate = int(fmt["sampleRate"])
    out_enc, out_rate = q.get("out_enc") or fmt["encoding"], int(q.get("out_rate") or in_rate)
    serializer = Logged(
        stream_id=start["start"]["streamId"],
        call_id=start["start"]["callId"],
        params=VobizFrameSerializer.InputParams(
            auto_hang_up=False,
            encoding=fmt["encoding"],
            vobiz_sample_rate=in_rate,
            output_encoding=out_enc,
            output_sample_rate=out_rate,
        ),
    )
    serializer.name_ = q.get("name", "x")
    serializer.swap = q.get("swap") == "1"
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, serializer=serializer
        ),
    )
    script = Script()
    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            script,
            transport.output(),
        ]
    )
    worker = PipelineWorker(
        pipeline, params=PipelineParams(audio_in_sample_rate=in_rate, audio_out_sample_rate=out_rate)
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def connected(t, c):
        asyncio.create_task(script.play(GREETING))

    @transport.event_handler("on_client_disconnected")
    async def disconnected(t, c):
        log("websocket closed", play_audio=serializer.played)
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
