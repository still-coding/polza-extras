import base64
import json
from decimal import Decimal

import pytest

from polza_extras import (
    ImageResult,
    InsufficientFundsError,
    MediaFailedError,
    MediaModeratedError,
    MediaTimeoutError,
    NotFoundError,
)
from tests.conftest import BASE
from tests.fake_server import ConnectError, resp

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 16
MODEL = "google/gemini-2.5-flash-image"
FILE = "https://cdn.example/out.png"


def job(status="completed", id="aig_1", url=FILE, cost="2.9", error=None, data=None):
    body = {"id": id, "status": status}
    if status == "completed":
        body["data"] = [{"url": url}] if data is None else data
        body["usage"] = {"cost_rub": cost}
    if error:
        body["error"] = error
    return body


@pytest.fixture
def media(server):
    return server.post(f"{BASE}/media")


@pytest.fixture
def status(server):
    return server.get(f"{BASE}/media/aig_1")


@pytest.fixture
def file(server):
    return server.get(FILE).respond(200, headers={"content-type": "image/png"})


def sent(route):
    return json.loads(route.calls[0].content)


def test_a_job_is_started_async_then_polled_and_the_file_downloaded(
    polza, media, status, file, sleeps
):
    media.respond(200, job("pending"))
    status.side_effect = [resp(200, job("processing")), resp(200, job("completed"))]
    file.respond(200, content=PNG)
    result = polza.generate_image(MODEL, "a cat", aspect_ratio="1:1", poll_interval=4.0)
    assert sent(media) == {
        "model": MODEL,
        "async": True,
        "input": {"prompt": "a cat", "aspect_ratio": "1:1"},
    }
    assert sleeps == [4.0, 4.0] and status.call_count == 2
    assert isinstance(result, ImageResult) and result.id == "aig_1" and result.url == FILE
    assert result.data == PNG
    assert result.cost_rub == Decimal("2.9") and result.model == MODEL


@pytest.mark.parametrize(("content", "mime"), [(PNG, "image/png"), (JPEG, "image/jpeg")])
def test_the_file_is_downloaded_and_its_type_is_sniffed(polza, media, file, content, mime):
    media.respond(200, job())
    file.respond(200, content=content)
    result = polza.generate_image(MODEL, "x")
    assert result.data == content and result.mime == mime
    assert "authorization" not in file.last_request.headers  # the key stays with Polza


def test_a_base64_answer_is_decoded_without_a_download(polza, media):
    media.respond(200, job(data=[{"b64_json": base64.b64encode(PNG).decode()}]))
    result = polza.generate_image(MODEL, "a cat")
    assert result.data == PNG and result.mime == "image/png" and result.url is None


def test_download_false_returns_the_url_only(polza, media, file):
    media.respond(200, job())
    result = polza.generate_image(MODEL, "a cat", download=False)
    assert result.data is None and result.url == FILE and file.call_count == 0


def test_reference_images_and_options_go_into_the_input(polza, media, tmp_path):
    media.respond(200, job(data=[{"b64_json": base64.b64encode(JPEG).decode()}]))
    path = tmp_path / "ref.png"
    path.write_bytes(PNG)
    polza.generate_image(
        MODEL,
        "same woman, smiling",
        images=[PNG, "https://cdn.example/ref.png", path],
        seed=7,
        image_resolution="1K",
        output_format="jpeg",
        enable_safety_checker=True,
    )
    body = sent(media)["input"]
    assert body["images"] == [
        {"type": "base64", "data": base64.b64encode(PNG).decode()},
        {"type": "url", "url": "https://cdn.example/ref.png"},
        {"type": "base64", "data": base64.b64encode(PNG).decode()},
    ]
    assert body["seed"] == 7 and body["image_resolution"] == "1K"
    assert body["output_format"] == "jpeg" and body["enable_safety_checker"] is True


def test_more_than_eight_references_are_refused_before_any_request(polza, media):
    with pytest.raises(ValueError, match="8"):
        polza.generate_image(MODEL, "x", images=[PNG] * 9)
    assert media.call_count == 0


def test_starting_the_job_is_retried_like_chat(polza, media, sleeps):
    media.side_effect = [
        resp(503, {"error": {"code": "SERVICE_UNAVAILABLE", "message": "m"}}),
        resp(200, job(data=[{"b64_json": base64.b64encode(PNG).decode()}])),
    ]
    assert polza.generate_image(MODEL, "x").data == PNG
    assert len(sleeps) == 1


def test_a_poll_error_is_retried_and_the_job_is_not_resent(polza, media, status, sleeps):
    media.respond(200, job("pending"))
    status.side_effect = [
        resp(502, {"error": {"code": "BAD_GATEWAY", "message": "m"}}),
        resp(200, job(data=[{"b64_json": base64.b64encode(PNG).decode()}])),
    ]
    assert polza.generate_image(MODEL, "x").data == PNG
    assert media.call_count == 1 and status.call_count == 2


def test_a_402_is_not_retried(polza, media):
    media.respond(402, {"error": {"code": "INSUFFICIENT_BALANCE", "message": "m"}})
    with pytest.raises(InsufficientFundsError):
        polza.generate_image(MODEL, "x")
    assert media.call_count == 1


def test_a_failed_job_raises_with_the_provider_reason(polza, media, status):
    media.respond(200, job("pending"))
    status.respond(
        200,
        job(
            "failed",
            error={
                "code": "BAD_GATEWAY",
                "message": "плохой ответ",
                "metadata": {"raw": "upstream 500", "provider_name": "google"},
            },
        ),
    )
    with pytest.raises(MediaFailedError) as caught:
        polza.generate_image(MODEL, "x")
    error = caught.value
    assert error.id == "aig_1" and error.code == "BAD_GATEWAY" and error.retryable
    assert error.provider == "google" and "upstream 500" in str(error)
    assert not isinstance(error, MediaModeratedError)


@pytest.mark.parametrize("code", ["guard_blocked", "guard_no_route"])
def test_moderation_is_its_own_error_and_not_retryable(polza, media, code):
    media.respond(
        200,
        job("failed", error={"code": code, "message": "m", "metadata": {"raw": "content policy"}}),
    )
    with pytest.raises(MediaModeratedError) as caught:
        polza.generate_image(MODEL, "x")
    assert caught.value.code == code and not caught.value.retryable


def test_a_cancelled_job_is_a_failure_that_is_not_retryable(polza, media):
    media.respond(200, job("cancelled"))
    with pytest.raises(MediaFailedError) as caught:
        polza.generate_image(MODEL, "x")
    assert caught.value.code == "cancelled" and not caught.value.retryable


def test_a_job_that_takes_too_long_raises_with_its_id_for_resuming(polza, media, status, sleeps):
    media.respond(200, job("pending"))
    status.respond(200, job("processing"))
    with pytest.raises(MediaTimeoutError) as caught:
        polza.generate_image(MODEL, "x", poll_interval=4.0, timeout=10.0)
    assert caught.value.id == "aig_1" and caught.value.status == "processing"
    assert sleeps == [4.0, 4.0, 4.0] and media.call_count == 1
    status.respond(200, job(data=[{"b64_json": base64.b64encode(PNG).decode()}]))
    assert polza.wait_image("aig_1", model=MODEL).data == PNG  # the same job, not a new one
    assert media.call_count == 1


def test_a_failed_download_raises_and_a_server_hiccup_is_retried(polza, media, file, sleeps):
    media.respond(200, job())
    file.side_effect = [resp(502, {"error": {"message": "bad gateway"}}), resp(200, content=PNG)]
    assert polza.generate_image(MODEL, "x").data == PNG and len(sleeps) == 1
    file.side_effect = [resp(404, {"error": {"message": "gone"}})]
    with pytest.raises(NotFoundError, match="gone"):
        polza.generate_image(MODEL, "x")


def test_a_connection_error_on_download_is_retried(polza, media, file, sleeps):
    media.respond(200, job())
    file.side_effect = [ConnectError("boom"), resp(200, content=PNG)]
    assert polza.generate_image(MODEL, "x").data == PNG and len(sleeps) == 1


def test_an_answer_without_a_picture_is_an_error(polza, media):
    media.respond(200, job(data=[]))
    with pytest.raises(MediaFailedError, match="no image"):
        polza.generate_image(MODEL, "x")


@pytest.mark.asyncio
async def test_async_generate_image(apolza, media, status):
    media.respond(200, job("pending"))
    status.respond(200, job(data=[{"b64_json": base64.b64encode(PNG).decode()}]))
    result = await apolza.generate_image(MODEL, "a cat")
    assert result.data == PNG and result.cost_rub == Decimal("2.9")


async def test_async_download_poll_timeout_and_resume(apolza, media, status, file, sleeps):
    media.respond(200, job("pending"))
    status.respond(200, job("processing"))
    with pytest.raises(MediaTimeoutError) as caught:
        await apolza.generate_image(MODEL, "x", poll_interval=4.0, timeout=4.0)
    assert caught.value.id == "aig_1" and sleeps == [4.0]
    status.respond(200, job())
    file.side_effect = [ConnectError("boom"), resp(200, content=PNG)]
    result = await apolza.wait_image("aig_1", model=MODEL)
    assert result.data == PNG and result.model == MODEL and media.call_count == 1
    assert sleeps[0] == 4.0 and len(sleeps) == 2  # the poll pause, then the download retry


async def test_async_failures_and_hiccups(apolza, media, file):
    media.side_effect = [
        resp(503, {"error": {"code": "SERVICE_UNAVAILABLE", "message": "m"}}),
        resp(200, job(url=FILE)),
    ]
    file.respond(404, {"error": {"message": "gone"}})
    with pytest.raises(NotFoundError):
        await apolza.generate_image(MODEL, "x")
    media.respond(200, job("failed", error={"code": "guard_blocked", "message": "m"}))
    with pytest.raises(MediaModeratedError):
        await apolza.generate_image(MODEL, "x")


async def test_async_references_and_url_only(apolza, media):
    media.respond(200, job())
    result = await apolza.generate_image(MODEL, "x", images=[PNG], download=False)
    assert result.url == FILE and result.data is None
    assert sent(media)["input"]["images"][0]["type"] == "base64"


def test_webp_and_unknown_files_are_sniffed():
    from polza_extras.media import sniff_mime

    assert sniff_mime(b"RIFF\0\0\0\0WEBPVP8 ") == "image/webp"
    assert sniff_mime(b"GIF89a") is None


def test_the_error_text_names_the_job_and_the_upstream_reason():
    error = MediaFailedError(
        "aig_9", "BAD_GATEWAY", "плохой ответ", {"raw": "upstream 500"}, retryable=True
    )
    assert "BAD_GATEWAY" in str(error) and "upstream 500" in str(error) and error.id == "aig_9"
    assert "aig_9" in str(MediaTimeoutError("aig_9", "pending", 300))
