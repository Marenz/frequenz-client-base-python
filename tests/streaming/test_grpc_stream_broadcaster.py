# License: MIT
# Copyright © 2024 Frequenz Energy-as-a-Service GmbH

"""Tests for GrpcStreamBroadcaster class."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from unittest import mock

import grpc.aio
import pytest

from frequenz.client.base import retry, streaming


def _transformer(x: int) -> str:
    """Mock transformer."""
    return f"transformed_{x}"


@pytest.fixture
def receiver_ready_event() -> asyncio.Event:
    """Fixture for receiver ready event."""
    return asyncio.Event()


@pytest.fixture
def no_retry() -> mock.MagicMock:
    """Fixture for mocked, non-retrying retry strategy."""
    mock_retry = mock.MagicMock(spec=retry.Strategy)
    mock_retry.next_interval.return_value = None
    mock_retry.copy.return_value = mock_retry
    mock_retry.get_progress.return_value = "mock progress"
    return mock_retry


@pytest.fixture
async def ok_helper(
    no_retry: mock.MagicMock,  # pylint: disable=redefined-outer-name
    receiver_ready_event: asyncio.Event,  # pylint: disable=redefined-outer-name
) -> AsyncIterator[streaming.GrpcStreamBroadcaster[int, str]]:
    """Fixture for GrpcStreamBroadcaster."""

    async def asynciter(ready_event: asyncio.Event) -> AsyncIterator[int]:
        """Mock async iterator."""
        await ready_event.wait()
        for i in range(5):
            yield i
            await asyncio.sleep(0)  # Yield control to the event loop

    helper = streaming.GrpcStreamBroadcaster(
        stream_name="test_helper",
        stream_method=lambda: asynciter(receiver_ready_event),
        transform=_transformer,
        retry_strategy=no_retry,
    )
    yield helper
    await helper.stop()


class _ErroringAsyncIter(AsyncIterator[int]):
    """Async iterator that raises an error after a certain number of successes."""

    def __init__(
        self, error: Exception, ready_event: asyncio.Event, num_successes: int = 0
    ):
        self._error = error
        self._ready_event = ready_event
        self._num_successes = num_successes
        self._current = -1

    async def __anext__(self) -> int:
        self._current += 1
        await self._ready_event.wait()
        if self._current >= self._num_successes:
            raise self._error
        return self._current


async def test_streaming_success(
    ok_helper: streaming.GrpcStreamBroadcaster[
        int, str
    ],  # pylint: disable=redefined-outer-name
    no_retry: mock.MagicMock,  # pylint: disable=redefined-outer-name
    receiver_ready_event: asyncio.Event,  # pylint: disable=redefined-outer-name
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test streaming success."""
    caplog.set_level(logging.INFO)
    items: list[str] = []
    async with asyncio.timeout(1):
        receiver = ok_helper.new_receiver()
        receiver_ready_event.set()
        async for item in receiver:
            items.append(item)
    no_retry.next_interval.assert_called_once_with()
    assert items == [
        "transformed_0",
        "transformed_1",
        "transformed_2",
        "transformed_3",
        "transformed_4",
    ]
    assert caplog.record_tuples == [
        (
            "frequenz.client.base.streaming",
            logging.ERROR,
            "test_helper: connection ended, retry limit exceeded (mock progress), "
            "giving up. Stream exhausted.",
        )
    ]


class _NamedMagicMock(mock.MagicMock):
    """Mock with a name."""

    def __str__(self) -> str:
        return self._mock_name  # type: ignore

    def __repr__(self) -> str:
        return self._mock_name  # type: ignore


@pytest.mark.parametrize("successes", [0, 1, 5])
async def test_streaming_error(  # pylint: disable=too-many-arguments
    successes: int,
    no_retry: mock.MagicMock,  # pylint: disable=redefined-outer-name
    receiver_ready_event: asyncio.Event,  # pylint: disable=redefined-outer-name
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test streaming errors."""
    caplog.set_level(logging.INFO)

    error = grpc.aio.AioRpcError(
        code=_NamedMagicMock(name="mock grpc code"),
        initial_metadata=mock.MagicMock(),
        trailing_metadata=mock.MagicMock(),
        details="mock details",
        debug_error_string="mock debug_error_string",
    )

    helper = streaming.GrpcStreamBroadcaster(
        stream_name="test_helper",
        stream_method=lambda: _ErroringAsyncIter(
            error, receiver_ready_event, num_successes=successes
        ),
        transform=_transformer,
        retry_strategy=no_retry,
    )

    items: list[str] = []
    async with AsyncExitStack() as stack:
        stack.push_async_callback(helper.stop)

        receiver = helper.new_receiver()
        receiver_ready_event.set()
        async for item in receiver:
            items.append(item)

    no_retry.next_interval.assert_called_once_with()
    assert items == [f"transformed_{i}" for i in range(successes)]
    assert caplog.record_tuples == [
        (
            "frequenz.client.base.streaming",
            logging.INFO,
            "test_helper: starting to stream",
        ),
        (
            "frequenz.client.base.streaming",
            logging.ERROR,
            "test_helper: connection ended, retry limit exceeded (mock progress), "
            "giving up. Error: "
            "<AioRpcError of RPC that terminated with:\n"
            "\tstatus = mock grpc code\n"
            '\tdetails = "mock details"\n'
            '\tdebug_error_string = "mock debug_error_string"\n'
            ">.",
        ),
    ]


async def test_retry_next_interval_zero(  # pylint: disable=too-many-arguments
    receiver_ready_event: asyncio.Event,  # pylint: disable=redefined-outer-name
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test retry logic when next_interval returns 0."""
    caplog.set_level(logging.WARNING)
    error = grpc.aio.AioRpcError(
        code=_NamedMagicMock(name="mock grpcio code"),
        initial_metadata=mock.MagicMock(),
        trailing_metadata=mock.MagicMock(),
        details="mock details",
        debug_error_string="mock debug_error_string",
    )
    mock_retry = mock.MagicMock(spec=retry.Strategy)
    mock_retry.next_interval.side_effect = [0, None]
    mock_retry.copy.return_value = mock_retry
    mock_retry.get_progress.return_value = "mock progress"
    helper = streaming.GrpcStreamBroadcaster(
        stream_name="test_helper",
        stream_method=lambda: _ErroringAsyncIter(error, receiver_ready_event),
        transform=_transformer,
        retry_strategy=mock_retry,
    )

    items: list[str] = []
    async with AsyncExitStack() as stack:
        stack.push_async_callback(helper.stop)

        receiver = helper.new_receiver()
        receiver_ready_event.set()
        async for item in receiver:
            items.append(item)

    assert not items
    assert mock_retry.next_interval.mock_calls == [mock.call(), mock.call()]
    expected_error_str = (
        "<AioRpcError of RPC that terminated with:\n"
        "\tstatus = mock grpcio code\n"
        '\tdetails = "mock details"\n'
        '\tdebug_error_string = "mock debug_error_string"\n'
        ">"
    )
    assert caplog.record_tuples == [
        (
            "frequenz.client.base.streaming",
            logging.WARNING,
            "test_helper: connection ended, retrying mock progress in 0.000 "
            f"seconds. Error: {expected_error_str}.",
        ),
        (
            "frequenz.client.base.streaming",
            logging.ERROR,
            "test_helper: connection ended, retry limit exceeded (mock progress), "
            f"giving up. Error: {expected_error_str}.",
        ),
    ]


async def test_new_receiver_after_error(
    no_retry: mock.MagicMock,  # pylint: disable=redefined-outer-name
    receiver_ready_event: asyncio.Event,  # pylint: disable=redefined-outer-name
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that creating a new receiver after an error restarts the stream."""
    caplog.set_level(logging.INFO)
    error = grpc.aio.AioRpcError(
        code=_NamedMagicMock(name="mock grpc code"),
        initial_metadata=mock.MagicMock(),
        trailing_metadata=mock.MagicMock(),
        details="mock details",
        debug_error_string="mock debug_error_string",
    )
    # Use the no_retry strategy
    helper = streaming.GrpcStreamBroadcaster(
        stream_name="test_helper",
        stream_method=lambda: _ErroringAsyncIter(
            error, receiver_ready_event, num_successes=1
        ),
        transform=_transformer,
        retry_strategy=no_retry,
    )

    items: list[str] = []
    async with AsyncExitStack() as stack:
        stack.push_async_callback(helper.stop)

        receiver = helper.new_receiver()
        receiver_ready_event.set()
        # Consume the first item before the error occurs
        async for item in receiver:
            items.append(item)

        # Wait for the helper's task to complete
        assert helper._task
        await helper._task
        assert helper._task.done()

        # At this point, the stream has ended due to the error
        # Now, create a new receiver after the error
        with mock.patch.object(helper, "start", wraps=helper.start) as mock_start:
            receiver = helper.new_receiver()
            # Ensure that helper.start() is called when the channel is closed
            mock_start.assert_called_once()

        # Reset the event to allow the new stream to proceed
        receiver_ready_event.clear()
        receiver_ready_event.set()
        async for item in receiver:
            items.append(item)

    # Verify that items from both streams are collected
    assert items == ["transformed_0", "transformed_0"]

    # Optionally, verify the logging output
    expected_logs = [
        (
            "frequenz.client.base.streaming",
            logging.INFO,
            "test_helper: starting to stream",
        ),
        (
            "frequenz.client.base.streaming",
            logging.ERROR,
            "test_helper: connection ended, retry limit exceeded (mock progress), "
            "giving up. Error: "
            "<AioRpcError of RPC that terminated with:\n"
            "\tstatus = mock grpc code\n"
            '\tdetails = "mock details"\n'
            '\tdebug_error_string = "mock debug_error_string"\n'
            ">.",
        ),
        (
            "frequenz.client.base.streaming",
            logging.WARNING,
            "test_helper: stream has stopped, starting a new one.",
        ),
        (
            "frequenz.client.base.streaming",
            logging.INFO,
            "test_helper: starting to stream",
        ),
        (
            "frequenz.client.base.streaming",
            logging.ERROR,
            "test_helper: connection ended, retry limit exceeded (mock progress), "
            "giving up. Error: "
            "<AioRpcError of RPC that terminated with:\n"
            "\tstatus = mock grpc code\n"
            '\tdetails = "mock details"\n'
            '\tdebug_error_string = "mock debug_error_string"\n'
            ">.",
        ),
    ]
    assert caplog.record_tuples == expected_logs
