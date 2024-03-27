"""
Like Python's zipfile, but using mmap instead of a regular file.

Helpful resources:

http://web.archive.org/web/20210219055401/https://blog.yaakov.online/zip64-go-big-or-go-home/
http://web.archive.org/web/20210310084602/https://users.cs.jmu.edu/buchhofp/forensics/formats/pkzip.html
http://web.archive.org/web/20210225050454/https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
"""

import mmap
import struct
import zlib
from typing import IO, Dict, List, Optional, no_type_check


class ZipInfo:
    """Contains information about a file stored in a ParallelZipFile."""

    filename: str
    header_offset: int
    CRC: int
    compress_size: int
    file_size: int

    def __init__(
        self,
        filename: str,
        header_offset: int,
        CRC: int,
        compress_size: int,
        file_size: int,
    ) -> None:
        self.filename = filename
        self.header_offset = header_offset
        self.CRC = CRC
        self.compress_size = compress_size
        self.file_size = file_size

    def is_dir(self) -> bool:
        """Directories in a zip file should end with '/'."""
        return self.filename.endswith("/")


def _read_eocd_mmap(m: mmap.mmap) -> Dict[str, ZipInfo]:
    # Read end-of-central-directory (EOCD) from mmaped zipfile.

    # TODO Can zip64 EOCDs be larger?
    max_eocd_size = 22 + 65536
    end = m[-max_eocd_size:]

    # Scan backwards until EOCD signature is found
    # TODO this could fail if a comment contains the EOCD signature.
    # Should employ sanity check to verify that an actual EOCD was found.
    offset32 = end.rfind(b"\x50\x4b\x05\x06")
    assert offset32 >= 0
    eocd = struct.unpack("<IHHHHIIH", end[offset32 : offset32 + 22])
    (
        signature,
        num_disks,
        num_disks2,
        num_files,
        num_files2,
        directory_size,
        directory_offset,
        comment_length,
    ) = eocd
    assert signature == 0x06054B50

    # If format is zip64, there should also be a zip64 EOCD header
    if num_files == 0xFFFF or directory_offset == 0xFFFFFFFF:
        offset64 = end.rfind(b"\x50\x4b\x06\x06")
        assert offset64 >= 0
        eocd = struct.unpack("<IQHHII4Q", end[offset64 : offset64 + 56])
        (
            signature,
            eocd_size,
            version,
            min_version,
            num_disks,
            num_disks2,
            num_files,
            num_files2,
            directory_size,
            directory_offset,
        ) = eocd
        assert signature == 0x06064B50

    # Read central directory headers which hold information about stored files
    files: Dict[str, ZipInfo] = {}
    mmap_offset = directory_offset
    for _ in range(num_files):
        header = m[mmap_offset : mmap_offset + 46]
        mmap_offset += 46

        (
            signature,
            version,
            min_version,
            unused0,
            compression,
            time,
            date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            unused1,
            attributes0,
            attributes1,
            offset,
        ) = struct.unpack("<I6H3I5HII", header)

        assert signature == 0x02014B50

        filename_bytes = m[mmap_offset : mmap_offset + filename_length]
        mmap_offset += filename_length
        extra = m[mmap_offset : mmap_offset + extra_length]
        mmap_offset += extra_length + comment_length
        filename = filename_bytes.rstrip(b"\0").decode("utf-8")

        # TODO Figure out what those bytes mean.
        # TODO Parse extra header correctly
        extra = extra[4:]

        if uncompressed_size == 0xFFFFFFFF:
            assert len(extra) >= 8
            uncompressed_size = struct.unpack("<Q", extra[:8])[0]
            extra = extra[8:]

        if compressed_size == 0xFFFFFFFF:
            assert len(extra) >= 8
            compressed_size = struct.unpack("<Q", extra[:8])[0]
            extra = extra[8:]

        if offset == 0xFFFFFFFF:
            assert len(extra) >= 8
            offset = struct.unpack("<Q", extra[-8:])[0]
            extra = extra[8:]

        info = ZipInfo(
            filename,
            offset,
            crc32,
            compressed_size,
            uncompressed_size,
        )

        files[filename] = info

    return files


def read_files(filename: str) -> Dict[str, ZipInfo]:
    """Read ZipInfo from zip file given its file path."""
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            return _read_eocd_mmap(m)


class ParallelZipFile:
    """Like Python's zipfile.ZipFile, but uses mmap instead of a file object for
    reading."""

    filename: str
    files: Dict[str, ZipInfo]
    f: IO[bytes]
    m: mmap.mmap

    def __init__(
        self, file: str, mode: str = "r", files: Optional[Dict[str, ZipInfo]] = None
    ) -> None:
        assert mode == "r"

        f = open(file, "rb")
        m = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

        if files is None:
            files = _read_eocd_mmap(m)

        self.filename: str = file
        self.files = files
        self.f = f
        self.m = m

    def __contains__(self, filename: str) -> bool:
        return filename in self.files

    def __enter__(self) -> "ParallelZipFile":
        return self

    @no_type_check
    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close internal file and mmap objects. Will be called automatically
        when using context manager, i.e. "with" statement."""
        if not isinstance(self.m, bytes):
            self.m.close()
        self.f.close()

    def namelist(self) -> List[str]:
        """Get file names for each file stored in zip file."""
        return list(self.files.keys())

    def infolist(self) -> List[ZipInfo]:
        """Get list of ZipInfo objects for each file stored in zip file."""
        return list(self.files.values())

    def read(self, filename: str) -> bytes:
        """Get bytes for file stored in zip file given its filename."""
        files = self.files

        if filename not in files:
            raise ValueError(f"{filename} does not exist")

        fileinfo = files[filename]

        offset = fileinfo.header_offset

        m = self.m

        header = m[offset : offset + 30]

        if len(header) < 30:
            error_message = f"Header for {filename} too small ({len(header)} bytes, but must be at least 30 bytes)"
            raise ValueError(error_message)

        (
            signature,
            version,
            unused,
            compression,
            time,
            date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
        ) = struct.unpack("<IHHHHHIIIHH", header)

        # TODO Is this legal/the indented way to do this?
        if compressed_size != fileinfo.compress_size:
            compressed_size = fileinfo.compress_size

        offset += 30 + filename_length + extra_length

        compressed = m[offset : offset + compressed_size]
        assert signature == 0x4034B50

        if compression == 0:
            # No compression
            return compressed
        elif compression == 8:
            # DEFLATE compression
            decompress = zlib.decompressobj(-zlib.MAX_WBITS)
            return decompress.decompress(compressed)
        else:
            error_message = f"Compression method {compression} not implemented"
            raise NotImplementedError(error_message)