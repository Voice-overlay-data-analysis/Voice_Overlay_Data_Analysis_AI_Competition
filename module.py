import struct
import binascii

def ConvertWav(data, endian):
    if endian == 'little':
        if len(data) == 4:
            data = struct.pack('<I', int(binascii.b2a_hex(data), 16))
        else:
            data = struct.pack('<h', int(binascii.b2a_hex(data), 16))
    return binascii.b2a_hex(data)