import os
import zlib
import numpy as np
from gnuradio import gr

# ── Golay(24,12) constants ────────────────────────────────────────────────────
_G_N = 12
_G_H = (0x8008ed, 0x4001db, 0x2003b5, 0x100769, 0x80ed1, 0x40da3,
        0x20b47,  0x1068f,  0x8d1d,   0x4a3b,   0x2477,  0x1ffe)


def _parity(v: int) -> int:
    v ^= v >> 16; v ^= v >> 8; v ^= v >> 4
    return (0x6996 >> (v & 0xF)) & 1


def _popcount(i: int) -> int:
    i -= (i >> 1) & 0x55555555
    i  = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


class GolayUncorrectable(Exception):
    pass


def _golay24_decode(word: int) -> tuple[int, int]:
    """Decode a 24-bit Golay codeword. Returns (data12, n_errors). Raises GolayUncorrectable."""
    def syndrome(w):
        return sum(_parity(_G_H[i] & w) << (_G_N - 1 - i) for i in range(_G_N))

    s = syndrome(word)

    if _popcount(s) <= 3:
        e = s << _G_N
    else:
        e = next(
            ((s ^ (_G_H[i] & 0xFFF)) << _G_N | (1 << (_G_N - i - 1))
             for i in range(_G_N) if _popcount(s ^ (_G_H[i] & 0xFFF)) <= 2),
            None
        )
        if e is None:
            q = syndrome(s)  # second syndrome
            if _popcount(q) <= 3:
                e = q
            else:
                e = next(
                    ((1 << (2 * _G_N - i - 1)) | (q ^ (_G_H[i] & 0xFFF))
                     for i in range(_G_N) if _popcount(q ^ (_G_H[i] & 0xFFF)) <= 2),
                    None
                )
                if e is None:
                    raise GolayUncorrectable()

    corrected = (word ^ e) & 0xFFF
    return corrected, _popcount(e)


# ── GNU Radio block ───────────────────────────────────────────────────────────

class blk(gr.sync_block):
    """
    Skylink / FORESAIL-1 frame decoder

    Input  : unpacked bits (uint8, one bit per byte, MSB-first)
    Output : none  (frames written to <out_dir>/clean_frames.bin)

    Framing: ASM (0x1ACFFC1D, 32 bits)
           + Golay(24,12) length field
           + RS(255,223) CCSDS-whitened codeword

    Output file is self-delimiting: each frame prefixed with 2-byte big-endian length.
    """

    ASM         = 0x1ACFFC1D
    ASM_LEN     = 32
    GOLAY_LEN   = 24
    CW_BYTES    = 255
    RS_BYTES    = 32
    DAT_BYTES   = 223
    ASM_MAX_ERR = 3
    MAGIC       = b'fOH2F1S'

    def __init__(self, out_dir="/tmp/skylink_out"):
        gr.sync_block.__init__(self, name="Skylink Frame Decoder",
                               in_sig=[np.uint8], out_sig=[])
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self._asm_bits = [(self.ASM >> b) & 1 for b in reversed(range(32))]
        self._cw_bits  = self.CW_BYTES * 8
        self._ks       = self._build_keystream(self.CW_BYTES)

        self._buf        = []
        self._abs0       = 0
        self._search_abs = 0
        self._asm_count  = 0
        self._seen       = set()
        self._clean_frames = []

    @staticmethod
    def _build_keystream(nbytes: int) -> bytes:
        """CCSDS PN sequence: poly x^8+x^7+x^5+x^3+1, init=0xFF."""
        reg, out = 0xFF, bytearray(nbytes)
        for bi in range(nbytes):
            v = 0
            for _ in range(8):
                v   = (v << 1) | (reg >> 7)
                fb  = ((reg >> 7) ^ (reg >> 4) ^ (reg >> 2) ^ reg) & 1
                reg = ((reg << 1) & 0xFF) | fb
            out[bi] = v
        return bytes(out)

    def _dewhiten(self, raw: bytes) -> bytes:
        return bytes(b ^ k for b, k in zip(raw, self._ks))

    def _asm_distance(self, bits) -> int:
        return sum(a != b for a, b in zip(bits, self._asm_bits))

    def _find_asm(self, start_rel: int) -> int:
        limit = len(self._buf) - (self.ASM_LEN + self.GOLAY_LEN + self._cw_bits)
        for i in range(start_rel, limit):
            if self._asm_distance(self._buf[i: i + 32]) <= self.ASM_MAX_ERR:
                return i
        return -1

    @staticmethod
    def _pack_msb(bits) -> bytes:
        out = bytearray(len(bits) // 8)
        for j, i in enumerate(range(0, len(bits), 8)):
            b = bits[i:i+8]
            out[j] = (b[0]<<7)|(b[1]<<6)|(b[2]<<5)|(b[3]<<4)|(b[4]<<3)|(b[5]<<2)|(b[6]<<1)|b[7]
        return bytes(out)

    @staticmethod
    def _bits_to_int(bits) -> int:
        v = 0
        for b in bits:
            v = (v << 1) | (b & 1)
        return v

    def _trim_buf(self):
        if len(self._buf) > 300_000:
            drop = len(self._buf) - 300_000
            del self._buf[:drop]
            self._abs0 += drop
            self._search_abs = max(self._search_abs, self._abs0)

    def _emit(self, de255: bytes, golay_raw: int, data12: int, n_errors: int, asm_abs: int):
        data_len    = data12 - self.RS_BYTES
        proto_id    = de255[0]
        sat_id      = de255[1:7]
        flags_vc    = de255[7]
        vc          = flags_vc & 0x07
        has_auth    = (flags_vc >> 3) & 1
        arq_on      = (flags_vc >> 4) & 1
        has_payload = (flags_vc >> 5) & 1
        ext_len     = de255[8]
        seq         = (de255[9] << 8) | de255[10]
        ext_end     = 11 + ext_len

        clean_frame = de255[:data_len]
        clean_ext   = clean_frame[11:min(ext_end, data_len)]
        after_ext   = clean_frame[ext_end:] if data_len > ext_end else b""
        clean_auth, clean_payload = (after_ext[-8:], after_ext[:-8]) \
            if has_auth and len(after_ext) >= 8 else (b"", after_ext)

        SEP = "─" * 60
        self._asm_count += 1
        print(f"\n{'═'*60}")
        print(f"  ASM #{self._asm_count}  @ bit {asm_abs}")
        print(SEP)
        print(f"  Protocol ID : 0x{proto_id:02X}  ({'OK' if proto_id == 0x66 else 'BAD'})")
        print(f"  Satellite   : {sat_id.decode(errors='replace')}  ({sat_id.hex()})")
        print(f"  Flags+VC    : 0x{flags_vc:02X}  "
              f"[HAS_PAYLOAD={has_payload} ARQ_ON={arq_on} HAS_AUTH={has_auth} VC={vc}]")
        print(f"  Frame seq   : {seq}  (0x{seq:04X})")
        print(f"  Ext hdr len : {ext_len}")
        if clean_ext:     print(f"  Ext header  : {clean_ext.hex()}")
        if clean_payload:
            print(f"  Payload     : {len(clean_payload)} bytes")
            for off in range(0, len(clean_payload), 16):
                chunk = clean_payload[off:off+16]
                h = " ".join(f"{b:02X}" for b in chunk)
                a = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                print(f"    [{off:03d}]  {h:<47}  {a}")
        if clean_auth:    print(f"  Auth (8B)   : {clean_auth.hex()}")
        print(f"  RS FEC (32B): {de255[self.DAT_BYTES:].hex()}")
        err_str = f"{n_errors} bit error(s) corrected" if n_errors else "no errors"
        print(f"  Golay       : 0x{golay_raw:06X}  data12={data12} → {data_len} clean bytes  [{err_str}]")
        print(f"  Clean frame : {clean_frame!r}")
        print(SEP)

        self._clean_frames.append(clean_frame)
        out_path = os.path.join(self.out_dir, "clean_frames.bin")
        with open(out_path, "wb") as f:
            for cf in self._clean_frames:
                f.write(len(cf).to_bytes(2, "big") + cf)
        print(f"  >> Wrote {len(self._clean_frames)} frame(s) to {out_path}", flush=True)

    def work(self, input_items, output_items):
        bits = input_items[0].astype(np.uint8) & 1
        if not len(bits):
            return 0

        self._buf.extend(bits.tolist())
        self._trim_buf()

        rel = max(0, self._search_abs - self._abs0)

        while True:
            asm_rel = self._find_asm(rel)
            if asm_rel < 0:
                break

            g_start = asm_rel + self.ASM_LEN
            c_start = g_start + self.GOLAY_LEN

            golay_bits = self._buf[g_start: g_start + self.GOLAY_LEN]
            cw_bits    = self._buf[c_start: c_start + self._cw_bits]

            if len(golay_bits) < self.GOLAY_LEN or len(cw_bits) < self._cw_bits:
                break

            def reject():
                nonlocal rel
                rel = asm_rel + 1
                self._search_abs = self._abs0 + rel

            golay_raw = self._bits_to_int(golay_bits)
            try:
                data12, n_errors = _golay24_decode(golay_raw)
            except GolayUncorrectable:
                reject(); continue

            data_len = data12 - self.RS_BYTES
            if not (1 <= data_len <= self.DAT_BYTES):
                reject(); continue

            raw255 = self._pack_msb(cw_bits)
            de255  = self._dewhiten(raw255)

            if de255[:7] != self.MAGIC or (11 + de255[8]) > self.DAT_BYTES:
                reject(); continue

            key = zlib.crc32(raw255) & 0xFFFFFFFF
            if key in self._seen:
                reject(); continue
            self._seen.add(key)

            self._emit(de255, golay_raw, data12, n_errors, self._abs0 + asm_rel)
            reject()

        return len(input_items[0])
