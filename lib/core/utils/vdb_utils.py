import uuid
import numpy
import collections

    
u8 = numpy.uint8
u16 = numpy.uint16
u32 = numpy.uint32
u64 = numpy.uint64
f16 = numpy.float16
f64 = numpy.float64


def bit_index(xyz, ratio, level):
    xyz = xyz & ((1 << (ratio + level)) - 1)
    redux = xyz >> ratio
    return redux[..., 0] | (redux[..., 1] << level) | (redux[..., 2] << (2 * level))


def write_lenstr(buffer: list, s: str):
    buffer.append(numpy.array([len(s)], u32))
    buffer.append(numpy.array(list(s.encode('ascii')), u8))


def getlen(buffer: list):
    return sum(map(lambda x: x.nbytes, buffer))


def coo_to_mask(nelem: int, coo: list):
    mask = numpy.zeros(nelem, dtype=numpy.bool8)
    mask[coo] = True
    return numpy.packbits(mask, bitorder='little')


def coo_to_dense(nelem: int, coo: list, vals: list):
    dense = numpy.zeros(nelem, dtype=f16)
    dense[coo] = vals
    return dense


def write_inter_node(buffer: list, node: dict, nelem: int):
    occmask = coo_to_mask(nelem, list(node.keys()))
    buffer.append(occmask)
    buffer.append(numpy.zeros_like(occmask))  # value mask
    buffer.append(numpy.array([6], u8))  # compression
    buffer.append(numpy.zeros([nelem], u16))


def dumps(density: numpy.ndarray, sparse_threshold: float = 0.01):

    def nestable_dict():
        return collections.defaultdict(nestable_dict)

    # density: [V, V, V], X-Y-Z indexing
    x, y, z = numpy.nonzero(density > sparse_threshold)
    xyz = numpy.stack([x, y, z], -1)
    bi4 = bit_index(xyz, 7, 5)
    bi3 = bit_index(xyz, 3, 4)
    bi0 = bit_index(xyz, 0, 3)
    root = nestable_dict()
    for sx, sy, sz, b4, b3, b0 in zip(x, y, z, bi4, bi3, bi0):
        root[b4][b3][b0] = density[sx, sy, sz]

    buffer = [numpy.array([0x20, 0x42, 0x44, 0x56, 0x0, 0x0, 0x0, 0x0], u8)]  # magic

    buffer.append(numpy.array([224, 8, 1], u32))  # version
    buffer.append(numpy.array([0], u8))  # grid offset
    buffer.append(numpy.array(list(str(uuid.uuid4()).encode('ascii')), u8))  # uuid

    buffer.append(numpy.array([0, 1], u32))  # no metadata, one grid

    write_lenstr(buffer, "density")
    write_lenstr(buffer, "Tree_float_5_4_3_HalfFloat")
    buffer.append(numpy.array([0], u32))  # no instancing
    buffer.append(numpy.array([getlen(buffer) + 3 * 8, 0, 0], u64))  # abs ofs
    buffer.append(numpy.array([0], u32))  # no compression

    buffer.append(numpy.array([4], u32))  # meta entries

    write_lenstr(buffer, "class")
    write_lenstr(buffer, "string")
    write_lenstr(buffer, "unknown")

    write_lenstr(buffer, "file_compression")
    write_lenstr(buffer, "string")
    write_lenstr(buffer, "none")

    write_lenstr(buffer, "is_saved_as_half_float")
    write_lenstr(buffer, "bool")
    buffer.append(numpy.array([1], u32))
    buffer.append(numpy.array([1], u8))
    
    write_lenstr(buffer, "name")
    write_lenstr(buffer, "string")
    write_lenstr(buffer, "density")

    write_lenstr(buffer, "AffineMap")
    buffer.append(numpy.eye(4, dtype=f64).reshape(-1))

    buffer.append(numpy.array([1, 0, 0, 1], u32))  # root 5-node
    buffer.append(numpy.array([0, 0, 0], u32))  # origin
    write_inter_node(buffer, root, 32768)
    for k4 in sorted(root.keys()):
        v4: dict = root[k4]
        write_inter_node(buffer, v4, 4096)
        for k3 in sorted(v4.keys()):
            v3: dict = v4[k3]
            buffer.append(coo_to_mask(512, list(v3.keys())))
    for k4 in sorted(root.keys()):
        v4: dict = root[k4]
        for k3 in sorted(v4.keys()):
            v3: dict = v4[k3]
            buffer.append(coo_to_mask(512, list(v3.keys())))
            buffer.append(numpy.array([6], u8))  # compression
            coo = sorted(v3.keys())
            stage = [v3[k0] for k0 in coo]
            buffer.append(coo_to_dense(512, coo, stage))
    return b''.join([x.tobytes() for x in buffer])
