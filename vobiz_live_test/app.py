"""Keyless live test for VobizFrameSerializer: pre-recorded speech and local Silero VAD.

Answers a real Vobiz call, plays a long greeting, and when the far end talks over
it, sends an interruption and plays a short reply. /events shows what crossed the wire.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
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
T0 = time.monotonic()
app = FastAPI()


def log(what: str, **extra) -> None:
    EVENTS.append({"t": round(time.monotonic() - T0, 3), "what": what, **extra})
    print(EVENTS[-1], flush=True)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/events")
async def events() -> list[dict]:
    return EVENTS


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request) -> Response:
    log("answer url requested")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
        f"wss://{request.headers['host']}/ws</Stream></Response>"
    )
    return Response(content=xml, media_type="application/xml")


class Logged(VobizFrameSerializer):
    played = 0

    async def serialize(self, frame):
        out = await super().serialize(frame)
        if out:
            event = json.loads(out).get("event")
            if event == "playAudio":
                self.played += 1
                if self.played == 1:
                    log("first playAudio sent")
            else:
                log(f"sent {event}", play_audio_so_far=self.played)
        return out

    async def deserialize(self, data):
        msg = json.loads(data)
        if msg.get("event") != "media":
            log(f"received {msg.get('event')}", raw=str(msg)[:160])
        frame = await super().deserialize(data)
        if frame is not None and msg.get("event") == "dtmf":
            log("InputDTMFFrame", digit=frame.button.value)
        return frame


class Script(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._interrupted = False
        self._replied = False

    async def play(self, clip: bytes) -> None:
        for i in range(0, len(clip), 320):
            await self.push_frame(
                OutputAudioRawFrame(audio=clip[i : i + 320], sample_rate=8000, num_channels=1)
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
    serializer = Logged(
        stream_id=start["start"]["streamId"],
        call_id=start["start"]["callId"],
        params=VobizFrameSerializer.InputParams(auto_hang_up=False),
    )
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
        pipeline, params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000)
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
