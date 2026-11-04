#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VobizFrameSerializer.

deserialize() ignores malformed JSON and messages missing expected fields rather
than raising, since the transport's receive loop ends the call on any exception.
With auto_hang_up enabled, the constructor requires call_id, auth_id and
auth_token, and the hang-up request authenticates with X-Auth headers.
"""

import base64
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
)
from pipecat.serializers.vobiz import VobizFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_ID = "c4dfd815-a92a-4140-ab85-5ff28c004116"
CALL_ID = "5401fd2e-6344-40df-a22c-c8ffea7a92e7"
AUTH_ID = "MA_TEST"
AUTH_TOKEN = "token"
SAMPLE_RATE = 8000


def _make_client_session(status: int = 204) -> MagicMock:
    response = MagicMock(status=status)
    response.text = AsyncMock(return_value="error")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.delete = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return session


class _SerializerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = VobizFrameSerializer(
            stream_id=STREAM_ID, params=VobizFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )


class TestVobizSerialize(_SerializerTestCase):
    async def test_audio_becomes_play_audio(self):
        message = json.loads(
            await self.serializer.serialize(
                OutputAudioRawFrame(
                    audio=b"\x00\x10" * 160, sample_rate=SAMPLE_RATE, num_channels=1
                )
            )
        )
        self.assertEqual(message["event"], "playAudio")
        self.assertEqual(message["streamId"], STREAM_ID)
        self.assertEqual(message["media"]["contentType"], "audio/x-mulaw")
        self.assertEqual(message["media"]["sampleRate"], SAMPLE_RATE)
        self.assertEqual(len(base64.b64decode(message["media"]["payload"])), 160)

    async def test_interruption_becomes_clear_audio(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"event": "clearAudio", "streamId": STREAM_ID})

    async def test_output_transport_message_is_passed_through(self):
        custom = {"event": "checkpoint", "streamId": STREAM_ID, "name": "response-1"}
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=custom))
        self.assertEqual(json.loads(result), custom)


class TestVobizDeserialize(_SerializerTestCase):
    async def test_media_becomes_input_audio(self):
        payload = base64.b64encode(b"\xff" * 160).decode()
        frame = await self.serializer.deserialize(
            json.dumps({"event": "media", "media": {"payload": payload}})
        )
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)
        self.assertEqual(len(frame.audio), 320)

    async def test_dtmf_becomes_input_dtmf(self):
        frame = await self.serializer.deserialize(
            json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})
        )
        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button.value, "5")

    async def test_invalid_messages_are_ignored_not_raised(self):
        for message in (
            "not json",
            json.dumps({"foo": "bar"}),
            json.dumps({"event": "media", "media": {}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "x"}}),
            json.dumps({"event": "clearedAudio", "streamId": STREAM_ID}),
            json.dumps({"event": "playedStream", "name": "response-1"}),
        ):
            with self.subTest(message=message):
                self.assertIsNone(await self.serializer.deserialize(message))


class TestVobizHangUp(unittest.IsolatedAsyncioTestCase):
    def _serializer(self) -> VobizFrameSerializer:
        return VobizFrameSerializer(
            stream_id=STREAM_ID, call_id=CALL_ID, auth_id=AUTH_ID, auth_token=AUTH_TOKEN
        )

    def test_auto_hang_up_requires_credentials(self):
        with self.assertRaisesRegex(ValueError, "call_id, auth_id, auth_token"):
            VobizFrameSerializer(stream_id=STREAM_ID)

    async def test_end_frame_deletes_call_with_auth_headers(self):
        session = _make_client_session()
        with patch("aiohttp.ClientSession", return_value=session):
            self.assertIsNone(await self._serializer().serialize(EndFrame()))
        session.delete.assert_called_once_with(
            f"https://api.vobiz.ai/api/v1/Account/{AUTH_ID}/Call/{CALL_ID}/",
            headers={"X-Auth-ID": AUTH_ID, "X-Auth-Token": AUTH_TOKEN},
        )

    async def test_hangs_up_only_once(self):
        session = _make_client_session()
        serializer = self._serializer()
        with patch("aiohttp.ClientSession", return_value=session):
            await serializer.serialize(EndFrame())
            await serializer.serialize(CancelFrame())
        session.delete.assert_called_once()

    async def test_failed_hang_up_does_not_raise(self):
        for status in (404, 500):
            with (
                self.subTest(status=status),
                patch("aiohttp.ClientSession", return_value=_make_client_session(status)),
            ):
                self.assertIsNone(await self._serializer().serialize(EndFrame()))
