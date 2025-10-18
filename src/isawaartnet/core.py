"""
isawaArtnet - Professional DMX to Art-Net Converter Library
Version: 0.1.1 (Art-Net 4 Compliant - FIXED)
Author: Based on pixcapture.py
License: MIT

This library provides complete Art-Net 4 protocol implementation with both
synchronous and asynchronous (asyncio) support for DMX512 data transmission.

CRITICAL: Art-Net 4 requires UNICAST transmission for ArtDmx packets.
Broadcast transmission is strictly prohibited by the specification.

FIXED: Length field now uses correct big-endian byte order
"""

import socket
import struct
import asyncio
from typing import List, Dict, Optional, Tuple, Any

__version__ = "1.0.0"
__all__ = [
    'DmxFrame', 'ArtNetPacket', 
    'ArtNetController', 'AsyncArtNetController',
    'ArtNetError'
]


class ArtNetError(Exception):
    """Exception raised for Art-Net specific errors"""
    pass


class DmxFrame:
    """
    DMX frame data container with validation and intuitive helper methods.
    
    PERFORMANCE: Internal data structure uses bytearray for maximum efficiency.
    Direct buffer access via .data property is the fastest method for
    performance-critical loops.
    
    CONVENIENCE: Helper methods provide intuitive channel setting for ease of use.
    
    Attributes:
        data (bytearray): Exactly 512 DMX channel values (0-255)
    
    Performance Example (fastest):
        >>> frame = DmxFrame.zeros()
        >>> buffer = frame.data
        >>> for i in range(1000):
        ...     buffer[0] = i % 256  # Direct access
        ...     artnet.send_dmx(frame)
    
    Convenience Example (intuitive):
        >>> frame = DmxFrame.zeros()
        >>> frame.set_channel(1, 255)
        >>> frame.set_channels(10, [100, 150, 200])
        >>> artnet.send_dmx(frame)
    """
    
    DMX_CHANNELS = 512
    CHANNEL_MIN = 1
    CHANNEL_MAX = 512
    VALUE_MIN = 0
    VALUE_MAX = 255
    
    def __init__(self, data: bytearray):
        if not isinstance(data, bytearray):
            raise TypeError(f"Data must be bytearray, got {type(data).__name__}")
        
        if len(data) != self.DMX_CHANNELS:
            raise ValueError(
                f"DMX frame must have exactly {self.DMX_CHANNELS} channels, "
                f"got {len(data)}"
            )
        
        self.data = data
    
    def set_channel(self, channel: int, value: int) -> None:
        """Set a single DMX channel value (1-512)."""
        if not self.CHANNEL_MIN <= channel <= self.CHANNEL_MAX:
            raise ValueError(
                f"Channel must be {self.CHANNEL_MIN}-{self.CHANNEL_MAX}, "
                f"got {channel}"
            )
        
        if not self.VALUE_MIN <= value <= self.VALUE_MAX:
            raise ValueError(
                f"Value must be {self.VALUE_MIN}-{self.VALUE_MAX}, "
                f"got {value}"
            )
        
        self.data[channel - 1] = value
    
    def set_channels(self, start_channel: int, values: List[int]) -> None:
        """Set multiple consecutive DMX channel values."""
        if not self.CHANNEL_MIN <= start_channel <= self.CHANNEL_MAX:
            raise ValueError(
                f"Start channel must be {self.CHANNEL_MIN}-{self.CHANNEL_MAX}, "
                f"got {start_channel}"
            )
        
        end_channel = start_channel + len(values) - 1
        if end_channel > self.CHANNEL_MAX:
            raise ValueError(
                f"Values extend beyond channel {self.CHANNEL_MAX} "
                f"(start: {start_channel}, count: {len(values)})"
            )
        
        # Pre-validate all values
        if not all(self.VALUE_MIN <= v <= self.VALUE_MAX for v in values):
            raise ValueError(
                f"All values must be {self.VALUE_MIN}-{self.VALUE_MAX}"
            )
        
        # Bulk assignment after validation
        for i, value in enumerate(values):
            self.data[start_channel - 1 + i] = value
    
    def set_from_dict(self, channel_dict: Dict[int, int]) -> None:
        """Set multiple channels from a dictionary {channel: value}."""
        for channel, value in channel_dict.items():
            self.set_channel(channel, value)
    
    def get_channel(self, channel: int) -> int:
        """Get the value of a single DMX channel (1-512)."""
        if not self.CHANNEL_MIN <= channel <= self.CHANNEL_MAX:
            raise ValueError(
                f"Channel must be {self.CHANNEL_MIN}-{self.CHANNEL_MAX}, "
                f"got {channel}"
            )
        
        return self.data[channel - 1]
    
    @classmethod
    def from_list(cls, data: List[int], pad_value: int = 0, 
                  strict: bool = True) -> 'DmxFrame':
        """
        Create DmxFrame from a list of values.
        
        Args:
            data: List of DMX values
            pad_value: Value to pad with if data < 512 channels
            strict: If True, raise error on invalid values. If False, clamp values.
        
        Raises:
            ValueError: If strict=True and values are out of range
        """
        if strict:
            # Validate all values
            for i, val in enumerate(data):
                if not (cls.VALUE_MIN <= val <= cls.VALUE_MAX):
                    raise ValueError(
                        f"Value at index {i} is {val}, must be "
                        f"{cls.VALUE_MIN}-{cls.VALUE_MAX}"
                    )
            normalized = bytearray(data)
        else:
            # Clamp values
            normalized = bytearray(
                max(cls.VALUE_MIN, min(cls.VALUE_MAX, int(val))) 
                for val in data
            )
        
        if len(normalized) > cls.DMX_CHANNELS:
            normalized = normalized[:cls.DMX_CHANNELS]
        elif len(normalized) < cls.DMX_CHANNELS:
            normalized.extend([pad_value] * (cls.DMX_CHANNELS - len(normalized)))
        
        return cls(normalized)
    
    @classmethod
    def zeros(cls) -> 'DmxFrame':
        """Create a DmxFrame with all channels at 0."""
        return cls(bytearray(cls.DMX_CHANNELS))
    
    @classmethod
    def full(cls, value: int = 255) -> 'DmxFrame':
        """Create a DmxFrame with all channels at specified value."""
        if not cls.VALUE_MIN <= value <= cls.VALUE_MAX:
            raise ValueError(
                f"Value must be {cls.VALUE_MIN}-{cls.VALUE_MAX}, got {value}"
            )
        return cls(bytearray([value] * cls.DMX_CHANNELS))


class ArtNetPacket:
    """Art-Net 4 protocol packet builder."""
    
    ARTNET_HEADER = b"Art-Net\x00"
    OPCODE_DMX = 0x5000
    OPCODE_POLL = 0x2000
    PROTOCOL_VERSION_HI = 0
    PROTOCOL_VERSION_LO = 14
    DMX_CHANNELS = 512
    
    TALK_TO_ME_SEND_REPLY = 0x02
    PRIORITY_DEFAULT = 0x00
    PHYSICAL_PORT_DEFAULT = 0x00
    
    SEQUENCE_MIN = 0  # Can be 0 to disable sequencing
    SEQUENCE_ENABLED_MIN = 1
    SEQUENCE_MAX = 255
    
    UNIVERSE_MIN = 0
    UNIVERSE_MAX = 32767
    
    @classmethod
    def create_dmx(cls, dmx_data: bytearray, universe: int, sequence: int,
                   physical: int = PHYSICAL_PORT_DEFAULT) -> bytearray:
        """
        Create an ArtDmx packet.
        
        Args:
            dmx_data: Exactly 512 bytes of DMX data
            universe: Universe number (0-32767)
            sequence: Sequence number (0=disabled, 1-255=enabled)
            physical: Physical port number (default=0)
        
        Returns:
            Complete ArtDmx packet ready for transmission
        
        Raises:
            ValueError: If parameters are out of range
        """
        if len(dmx_data) != cls.DMX_CHANNELS:
            raise ValueError(
                f"DMX data must have exactly {cls.DMX_CHANNELS} channels, "
                f"got {len(dmx_data)}"
            )
        
        if not cls.UNIVERSE_MIN <= universe <= cls.UNIVERSE_MAX:
            raise ValueError(
                f"Universe must be {cls.UNIVERSE_MIN}-{cls.UNIVERSE_MAX}, "
                f"got {universe}"
            )
        
        if not cls.SEQUENCE_MIN <= sequence <= cls.SEQUENCE_MAX:
            raise ValueError(
                f"Sequence must be {cls.SEQUENCE_MIN}-{cls.SEQUENCE_MAX}, "
                f"got {sequence}"
            )
        
        # 15-bit Port-Address encoding
        sub_uni = universe & 0xFF        # Low 8 bits (Sub-Net + Universe)
        net = (universe >> 8) & 0x7F     # High 7 bits (Net)
        
        # 🔧 FIX: Pack header with correct byte orders
        # OpCode: little-endian (< prefix)
        # Length: big-endian (> prefix)
        header = struct.pack(
            '<8s H BB BBBB',
            cls.ARTNET_HEADER,           # ID[8]
            cls.OPCODE_DMX,              # OpCode (little-endian)
            cls.PROTOCOL_VERSION_HI,     # ProtVerHi
            cls.PROTOCOL_VERSION_LO,     # ProtVerLo
            sequence,                     # Sequence
            physical,                     # Physical
            sub_uni,                      # SubUni
            net,                          # Net
        )
        
        # Add Length field separately as big-endian
        length_bytes = struct.pack('>H', cls.DMX_CHANNELS)  # Big-endian
        
        packet = bytearray(header)
        packet.extend(length_bytes)
        packet.extend(dmx_data)
        
        return packet
    
    @classmethod
    def create_poll(cls) -> bytearray:
        """Create an ArtPoll packet for device discovery."""
        packet = struct.pack(
            '<8s H BB BB',
            cls.ARTNET_HEADER,
            cls.OPCODE_POLL,
            cls.PROTOCOL_VERSION_HI,
            cls.PROTOCOL_VERSION_LO,
            cls.TALK_TO_ME_SEND_REPLY,
            cls.PRIORITY_DEFAULT
        )
        
        return bytearray(packet)


class ArtNetController:
    """
    Synchronous Art-Net 4 network controller.
    
    This controller implements Art-Net 4 specification with UNICAST transmission.
    Broadcast transmission is not supported as per Art-Net 4 requirements.
    
    IMPORTANT: You must specify a target IP address. Each ArtDmx packet is
    unicast to the specified device.
    
    Example:
        >>> frame = DmxFrame.zeros()
        >>> frame.set_channel(1, 255)
        >>> 
        >>> with ArtNetController("192.168.1.100") as artnet:
        ...     artnet.send_dmx(frame)
    """
    
    DEFAULT_PORT = 6454
    SOCKET_TIMEOUT = 1.0
    HEADER_SIZE = 18
    
    def __init__(self, target_ip: str, port: int = DEFAULT_PORT,
                 enable_sequence: bool = True):
        """
        Initialize Art-Net controller.
        
        Args:
            target_ip: Target device IP address (REQUIRED, unicast only)
            port: UDP port (default: 6454)
            enable_sequence: Enable sequence numbering (default: True)
        
        Raises:
            ValueError: If port is invalid or target_ip is broadcast
        """
        if not 1 <= port <= 65535:
            raise ValueError(f"Port must be 1-65535, got {port}")
        
        # Check for broadcast addresses (Art-Net 4 violation)
        if target_ip in ("255.255.255.255", "2.255.255.255", "10.255.255.255"):
            raise ValueError(
                f"Broadcast address '{target_ip}' is not allowed for ArtDmx. "
                f"Art-Net 4 requires unicast transmission. "
                f"Please specify the target device's IP address."
            )
        
        self.target_ip = target_ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.enable_sequence = enable_sequence
        self.sequence = ArtNetPacket.SEQUENCE_ENABLED_MIN if enable_sequence else 0
        self.is_connected = False
        self.debug_mode = False
        self.packet_count = 0
        
        self._init_socket()
    
    def _init_socket(self) -> None:
        """Initialize UDP socket."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.SOCKET_TIMEOUT)
            self.is_connected = True
            
            if self.debug_mode:
                print(f"✓ Socket initialized: {self.target_ip}:{self.port}")
        except OSError as e:
            raise ArtNetError(f"Failed to initialize socket: {e}") from e
    
    def enable_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug output."""
        self.debug_mode = enabled
        if enabled:
            print(f"Debug mode enabled - Target: {self.target_ip}:{self.port}")
    
    def _increment_sequence(self) -> None:
        """Increment sequence number (1-255 loop), or keep at 0 if disabled."""
        if self.enable_sequence:
            if self.sequence >= ArtNetPacket.SEQUENCE_MAX:
                self.sequence = ArtNetPacket.SEQUENCE_ENABLED_MIN
            else:
                self.sequence += 1
    
    def send_dmx(self, frame: DmxFrame, universe: int = 0,
                 physical: int = 0) -> None:
        """
        Send a DMX frame via Art-Net (unicast).
        
        Args:
            frame: DmxFrame object with exactly 512 channels
            universe: Universe number (0-32767, default: 0)
            physical: Physical port number (default: 0)
        
        Raises:
            ValueError: If universe is out of range
            ArtNetError: If network transmission fails
        """
        if not self.is_connected or not self.socket:
            self._init_socket()
        
        packet = ArtNetPacket.create_dmx(frame.data, universe, self.sequence, physical)
        
        try:
            bytes_sent = self.socket.sendto(packet, (self.target_ip, self.port))
        except OSError as e:
            self.is_connected = False
            raise ArtNetError(
                f"Failed to send to {self.target_ip}:{self.port} - {e}"
            ) from e
        
        self.packet_count += 1
        self._increment_sequence()
        
        expected_size = self.HEADER_SIZE + DmxFrame.DMX_CHANNELS
        if bytes_sent != expected_size:
            raise ArtNetError(
                f"Packet size mismatch: sent {bytes_sent}, expected {expected_size}"
            )
        
        if self.debug_mode:
            print(f"✓ Sent DMX to universe {universe}: {bytes_sent} bytes, seq={self.sequence-1}")
    
    def send_artpoll(self, broadcast_address: str = "2.255.255.255") -> None:
        """
        Send Art-Poll packet for device discovery.
        
        Note: ArtPoll CAN be broadcast (unlike ArtDmx).
        
        Args:
            broadcast_address: Broadcast address for discovery (default: 2.255.255.255)
        """
        if not self.is_connected or not self.socket:
            self._init_socket()
        
        packet = ArtNetPacket.create_poll()
        
        try:
            # Enable broadcast for this specific operation
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.sendto(packet, (broadcast_address, self.port))
            if self.debug_mode:
                print(f"✓ Art-Poll broadcast sent to {broadcast_address}")
        except OSError as e:
            self.is_connected = False
            raise ArtNetError(
                f"Failed to send Art-Poll to {broadcast_address}:{self.port} - {e}"
            ) from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transmission statistics."""
        return {
            'packets_sent': self.packet_count,
            'target_ip': self.target_ip,
            'port': self.port,
            'sequence': self.sequence,
            'sequence_enabled': self.enable_sequence,
            'connected': self.is_connected
        }
    
    def close(self) -> None:
        """Close socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None
                self.is_connected = False
                if self.debug_mode:
                    print("✓ Connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AsyncArtNetController:
    """
    Asynchronous Art-Net 4 network controller using asyncio.
    
    This controller implements Art-Net 4 specification with UNICAST transmission
    in a non-blocking manner, perfect for GUI applications and high-concurrency
    scenarios.
    
    IMPORTANT: You must specify a target IP address. Each ArtDmx packet is
    unicast to the specified device.
    
    Perfect for:
    - GUI applications (Tkinter, PyQt, CustomTkinter, etc.)
    - Web applications (FastAPI, aiohttp)
    - Concurrent network operations
    - Non-blocking DMX transmission
    
    Example:
        >>> async def main():
        ...     frame = DmxFrame.zeros()
        ...     frame.set_channel(1, 255)
        ...     
        ...     async with AsyncArtNetController("192.168.1.100") as artnet:
        ...         await artnet.send_dmx(frame)
        ...         await asyncio.sleep(0.02)
        >>> 
        >>> asyncio.run(main())
    """
    
    DEFAULT_PORT = 6454
    HEADER_SIZE = 18
    
    def __init__(self, target_ip: str, port: int = DEFAULT_PORT,
                 enable_sequence: bool = True):
        """
        Initialize async Art-Net controller.
        
        Args:
            target_ip: Target device IP address (REQUIRED, unicast only)
            port: UDP port (default: 6454)
            enable_sequence: Enable sequence numbering (default: True)
        
        Raises:
            ValueError: If port is invalid or target_ip is broadcast
        """
        if not 1 <= port <= 65535:
            raise ValueError(f"Port must be 1-65535, got {port}")
        
        # Check for broadcast addresses (Art-Net 4 violation)
        if target_ip in ("255.255.255.255", "2.255.255.255", "10.255.255.255"):
            raise ValueError(
                f"Broadcast address '{target_ip}' is not allowed for ArtDmx. "
                f"Art-Net 4 requires unicast transmission. "
                f"Please specify the target device's IP address."
            )
        
        self.target_ip = target_ip
        self.port = port
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None
        self.enable_sequence = enable_sequence
        self.sequence = ArtNetPacket.SEQUENCE_ENABLED_MIN if enable_sequence else 0
        self.is_connected = False
        self.debug_mode = False
        self.packet_count = 0
    
    async def _init_socket(self) -> None:
        """Initialize asynchronous UDP socket."""
        loop = asyncio.get_event_loop()
        
        class ArtNetProtocol(asyncio.DatagramProtocol):
            def __init__(self):
                self.transport = None
            
            def connection_made(self, transport):
                self.transport = transport
            
            def datagram_received(self, data, addr):
                pass  # We don't expect responses for ArtDmx
            
            def error_received(self, exc):
                if exc:
                    print(f"Protocol error: {exc}")
            
            def connection_lost(self, exc):
                pass
        
        try:
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                ArtNetProtocol,
                local_addr=('0.0.0.0', 0)
            )
            
            self.is_connected = True
            
            if self.debug_mode:
                print(f"✓ Async socket initialized: {self.target_ip}:{self.port}")
                
        except Exception as e:
            raise ArtNetError(f"Failed to initialize async socket: {e}") from e
    
    def enable_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug output."""
        self.debug_mode = enabled
        if enabled:
            print(f"Debug mode enabled (async) - Target: {self.target_ip}:{self.port}")
    
    def _increment_sequence(self) -> None:
        """Increment sequence number (1-255 loop), or keep at 0 if disabled."""
        if self.enable_sequence:
            if self.sequence >= ArtNetPacket.SEQUENCE_MAX:
                self.sequence = ArtNetPacket.SEQUENCE_ENABLED_MIN
            else:
                self.sequence += 1
    
    async def send_dmx(self, frame: DmxFrame, universe: int = 0,
                       physical: int = 0) -> None:
        """
        Send a DMX frame via Art-Net asynchronously (unicast).
        
        This method does not block the event loop, allowing other async
        tasks to run concurrently.
        
        Args:
            frame: DmxFrame object with exactly 512 channels
            universe: Universe number (0-32767, default: 0)
            physical: Physical port number (default: 0)
            
        Raises:
            ValueError: If universe is out of range
            ArtNetError: If network transmission fails
        """
        if not self.is_connected or not self.transport:
            await self._init_socket()
        
        packet = ArtNetPacket.create_dmx(frame.data, universe, self.sequence, physical)
        
        try:
            self.transport.sendto(packet, (self.target_ip, self.port))
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
            
        except Exception as e:
            self.is_connected = False
            raise ArtNetError(
                f"Failed to send to {self.target_ip}:{self.port} - {e}"
            ) from e
        
        self.packet_count += 1
        self._increment_sequence()
        
        if self.debug_mode:
            expected_size = self.HEADER_SIZE + DmxFrame.DMX_CHANNELS
            print(f"✓ Async sent DMX to universe {universe}: {expected_size} bytes, seq={self.sequence-1}")
    
    async def send_artpoll(self, broadcast_address: str = "2.255.255.255") -> None:
        """
        Send Art-Poll packet for device discovery asynchronously.
        
        Note: ArtPoll CAN be broadcast (unlike ArtDmx).
        
        Args:
            broadcast_address: Broadcast address for discovery (default: 2.255.255.255)
        
        Raises:
            ArtNetError: If transmission fails
        """
        if not self.is_connected or not self.transport:
            await self._init_socket()
        
        packet = ArtNetPacket.create_poll()
        
        try:
            self.transport.sendto(packet, (broadcast_address, self.port))
            await asyncio.sleep(0)
            
        except Exception as e:
            self.is_connected = False
            raise ArtNetError(
                f"Failed to send Art-Poll to {broadcast_address}:{self.port} - {e}"
            ) from e
        
        if self.debug_mode:
            print(f"✓ Async Art-Poll sent to {broadcast_address}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transmission statistics."""
        return {
            'packets_sent': self.packet_count,
            'target_ip': self.target_ip,
            'port': self.port,
            'sequence': self.sequence,
            'sequence_enabled': self.enable_sequence,
            'connected': self.is_connected
        }
    
    async def close(self) -> None:
        """Close async socket connection."""
        if self.transport:
            self.transport.close()
            await asyncio.sleep(0.1)  # Allow time for proper closure
            self.transport = None
            self.is_connected = False
            if self.debug_mode:
                print("✓ Async connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_socket()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False