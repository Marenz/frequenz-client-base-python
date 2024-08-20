# License: MIT
# Copyright © 2024 Frequenz Energy-as-a-Service GmbH

"""Base class for API clients."""

import abc
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Self, TypeVar, overload

from grpc.aio import AioRpcError, Channel

from .channel import ChannelOptions, parse_grpc_uri
from .exception import ApiClientError, ClientNotConnected

StubT = TypeVar("StubT")
"""The type of the gRPC stub."""


class BaseApiClient(abc.ABC, Generic[StubT]):
    """A base class for API clients.

    This class provides a common interface for API clients that communicate with a API
    server. It is designed to be subclassed by specific API clients that provide a more
    specific interface.

    Some extra tools are provided to make it easier to write API clients:

    - [call_stub_method()][frequenz.client.base.client.call_stub_method] is a function
        that calls a gRPC stub method and translates errors to API client errors.
    - [GrpcStreamBroadcaster][frequenz.client.base.streaming.GrpcStreamBroadcaster] is
        a class that helps sending messages from a gRPC stream to
        a [Broadcast][frequenz.channels.Broadcast] channel.

    Example:
        This example illustrates how to create a simple API client that connects to a
        gRPC server and calls a method on a stub.

        ```python
        from collections.abc import AsyncIterable
        from frequenz.client.base.client import BaseApiClient, call_stub_method
        from frequenz.client.base.streaming import GrpcStreamBroadcaster
        from frequenz.channels import Receiver

        # These classes are normally generated by protoc
        class ExampleRequest:
            int_value: int
            str_value: str

        class ExampleResponse:
            float_value: float

        class ExampleStub:
            async def example_method(
                self,
                request: ExampleRequest  # pylint: disable=unused-argument
            ) -> ExampleResponse:
                ...

            def example_stream(self) -> AsyncIterable[ExampleResponse]:
                ...
        # End of generated classes

        class ExampleResponseWrapper:
            def __init__(self, response: ExampleResponse):
                self.transformed_value = f"{response.float_value:.2f}"

        class MyApiClient(BaseApiClient[ExampleStub]):
            def __init__(self, server_url: str, *, connect: bool = True):
                super().__init__(
                    server_url, ExampleStub, connect=connect
                )
                self._broadcaster = GrpcStreamBroadcaster(
                    "stream",
                    lambda: self.stub.example_stream(ExampleRequest()),
                    ExampleResponseWrapper,
                )

            async def example_method(
                self, int_value: int, str_value: str
            ) -> ExampleResponseWrapper:
                return await call_stub_method(
                    self,
                    lambda: self.stub.example_method(
                        ExampleRequest(int_value=int_value, str_value=str_value)
                    ),
                    transform=ExampleResponseWrapper,
                )

            def example_stream(self) -> Receiver[ExampleResponseWrapper]:
                return self._broadcaster.new_receiver()


        async def main():
            client = MyApiClient("grpc://localhost")
            response = await client.example_method(42, "hello")
            print(response.transformed_value)
            count = 0
            async for response in client.example_stream():
                print(response.transformed_value)
                count += 1
                if count >= 5:
                    break
        ```

        Note:
            * In this case a very simple `GrpcStreamBroadcaster` is used, asuming that
                each call to `example_stream` will stream the same data. If the request
                is more complex, you will probably need to have some kind of map from
                a key based on the stream method request parameters to broadcaster
                instances.
    """

    def __init__(
        self,
        server_url: str,
        create_stub: Callable[[Channel], StubT],
        *,
        connect: bool = True,
        channel_defaults: ChannelOptions = ChannelOptions(),
    ) -> None:
        """Create an instance and connect to the server.

        Args:
            server_url: The URL of the server to connect to.
            create_stub: A function that creates a stub from a channel.
            connect: Whether to connect to the server as soon as a client instance is
                created. If `False`, the client will not connect to the server until
                [connect()][frequenz.client.base.client.BaseApiClient.connect] is
                called.
            channel_defaults: The default options for the gRPC channel to create using
                the server URL.
        """
        self._server_url: str = server_url
        self._create_stub: Callable[[Channel], StubT] = create_stub
        self._channel_defaults: ChannelOptions = channel_defaults
        self._channel: Channel | None = None
        self._stub: StubT | None = None
        if connect:
            self.connect(server_url)

    @property
    def server_url(self) -> str:
        """The URL of the server."""
        return self._server_url

    @property
    def channel(self) -> Channel:
        """The underlying gRPC channel used to communicate with the server.

        Warning:
            This channel is provided as a last resort for advanced users. It is not
            recommended to use this property directly unless you know what you are
            doing and you don't care about being tied to a specific gRPC library.

        Raises:
            ClientNotConnected: If the client is not connected to the server.
        """
        if self._channel is None:
            raise ClientNotConnected(server_url=self.server_url, operation="channel")
        return self._channel

    @property
    def channel_defaults(self) -> ChannelOptions:
        """The default options for the gRPC channel."""
        return self._channel_defaults

    @property
    def stub(self) -> StubT:
        """The underlying gRPC stub.

        Warning:
            This stub is provided as a last resort for advanced users. It is not
            recommended to use this property directly unless you know what you are
            doing and you don't care about being tied to a specific gRPC library.

        Raises:
            ClientNotConnected: If the client is not connected to the server.
        """
        if self._stub is None:
            raise ClientNotConnected(server_url=self.server_url, operation="stub")
        return self._stub

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to the server."""
        return self._channel is not None

    def connect(self, server_url: str | None = None) -> None:
        """Connect to the server, possibly using a new URL.

        If the client is already connected and the URL is the same as the previous URL,
        this method does nothing. If you want to force a reconnection, you can call
        [disconnect()][frequenz.client.base.client.BaseApiClient.disconnect] first.

        Args:
            server_url: The URL of the server to connect to. If not provided, the
                previously used URL is used.
        """
        if server_url is not None and server_url != self._server_url:  # URL changed
            self._server_url = server_url
        elif self.is_connected:
            return
        self._channel = parse_grpc_uri(self._server_url, self._channel_defaults)
        self._stub = self._create_stub(self._channel)

    async def disconnect(self) -> None:
        """Disconnect from the server.

        If the client is not connected, this method does nothing.
        """
        await self.__aexit__(None, None, None)

    async def __aenter__(self) -> Self:
        """Enter a context manager."""
        self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any | None,
    ) -> bool | None:
        """Exit a context manager."""
        if self._channel is None:
            return None
        result = await self._channel.__aexit__(_exc_type, _exc_val, _exc_tb)
        self._channel = None
        self._stub = None
        return result


StubOutT = TypeVar("StubOutT")
"""The type of the response from a gRPC stub method."""

TransformOutT_co = TypeVar("TransformOutT_co", covariant=True)
"""The type of the transformed response from a gRPC stub method."""


@overload
async def call_stub_method(
    client: BaseApiClient[StubT],
    stub_method: Callable[[], Awaitable[StubOutT]],
    *,
    method_name: str | None = None,
    transform: Callable[[StubOutT], TransformOutT_co],
) -> TransformOutT_co: ...


@overload
async def call_stub_method(
    client: BaseApiClient[StubT],
    stub_method: Callable[[], Awaitable[StubOutT]],
    *,
    method_name: str | None = None,
    transform: None = None,
) -> StubOutT: ...


async def call_stub_method(
    client: BaseApiClient[StubT],
    stub_method: Callable[[], Awaitable[StubOutT]],
    *,
    method_name: str | None = None,
    transform: Callable[[StubOutT], TransformOutT_co] | None = None,
) -> StubOutT | TransformOutT_co:
    """Call a gRPC stub method and translate errors to API client errors.

    This function is a convenience wrapper around calling a gRPC stub method. It
    translates gRPC errors to API client errors and optionally transforms the response
    using a provided function.

    This function is designed to be used with API clients that subclass
    [BaseApiClient][frequenz.client.base.client.BaseApiClient].

    Args:
        client: The API client to use.
        stub_method: The gRPC stub method to call.
        method_name: The name of the method being called. If not provided, the name of
            the calling function is used.
        transform: A function that transforms the response from the gRPC stub method.

    Returns:
        The response from the gRPC stub method, possibly transformed by the `transform`
            function if provided.

    Raises:
        ClientNotConnected: If the client is not connected to the server.
        GrpcError: If a gRPC error occurs.
    """
    if method_name is None:
        # Get the name of the calling function
        method_name = inspect.stack()[1][3]

    if not client.is_connected:
        raise ClientNotConnected(server_url=client.server_url, operation=method_name)

    try:
        response = await stub_method()
    except AioRpcError as grpc_error:
        raise ApiClientError.from_grpc_error(
            server_url=client.server_url, operation=method_name, grpc_error=grpc_error
        ) from grpc_error

    return response if transform is None else transform(response)
