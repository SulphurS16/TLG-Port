r"""Writes the application icon (res/app.ico): a pale feather
on a dark round field, drawn here (original artwork, no game assets).

    python -I tools/make_icon.py res/app.ico

Sizes 16, 24, 32, 48, 64 and 256 px as 32-bit BMP images (4x4 supersampled).
"""
import math
import struct
import sys


def shade(x, y):
    """RGBA of the design at (x, y) in [-1, 1]^2."""
    r = math.hypot(x, y)
    if r > 0.98:
        return (0, 0, 0, 0)
    # Field: dark teal, lighter towards the top left.
    t = max(0.0, min(1.0, 0.55 - 0.35 * (x + y)))
    base = (int(18 + 22 * t), int(44 + 30 * t), int(52 + 30 * t), 255)
    # Feather along the diagonal from bottom left to top right.
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    u = x * c - y * s     # along the feather (towards the top right is negative y)
    v = x * s + y * c
    u, v = (x * c + (-y) * s), (-x * s + (-y) * c)
    half = 0.78           # half length
    if -half < u < half:
        p = (u + half) / (2 * half)          # 0 at the quill, 1 at the tip
        width = 0.30 * math.sin(math.pi * min(1.0, p * 1.15)) ** 0.8 * (1.0 - 0.25 * p)
        if abs(v) < width:
            # Vane with a few barb lines; the rachis darker.
            barb = 0.5 + 0.5 * math.cos((u * 9.0 + abs(v) * 6.0) * math.pi)
            lum = 0.86 + 0.10 * barb
            col = (int(232 * lum), int(220 * lum), int(196 * lum), 255)
            if abs(v) < 0.022 + 0.01 * (1 - p):
                col = (150, 132, 110, 255)
            return col
    if -half - 0.18 < u <= -half and abs(v) < 0.02:
        return (150, 132, 110, 255)
    return base


def render(n):
    ss = 4
    px = []
    for j in range(n):
        row = []
        for i in range(n):
            acc = [0, 0, 0, 0]
            for b in range(ss):
                for a in range(ss):
                    x = ((i + (a + 0.5) / ss) / n) * 2 - 1
                    y = ((j + (b + 0.5) / ss) / n) * 2 - 1
                    r, g, bl, al = shade(x, y)
                    acc[0] += r * al
                    acc[1] += g * al
                    acc[2] += bl * al
                    acc[3] += al
            k = ss * ss
            al = acc[3] / k
            if acc[3]:
                row.append((acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], al))
            else:
                row.append((0, 0, 0, 0))
        px.append(row)
    return px


def bmp_image(n):
    px = render(n)
    header = struct.pack('<IiiHHIIiiII', 40, n, n * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    xor = bytearray()
    for row in reversed(px):
        for r, g, b, a in row:
            xor += bytes((int(b), int(g), int(r), int(a)))
    mask_row = ((n + 31) // 32) * 4
    andmask = bytes(mask_row * n)
    return header + bytes(xor) + andmask


def main():
    out = sys.argv[1]
    sizes = [16, 24, 32, 48, 64, 256]
    images = [bmp_image(n) for n in sizes]
    data = struct.pack('<HHH', 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    for n, img in zip(sizes, images):
        data += struct.pack('<BBBBHHII', n % 256, n % 256, 0, 0, 1, 32, len(img), offset)
        offset += len(img)
    for img in images:
        data += img
    with open(out, 'wb') as f:
        f.write(data)
    print(f'wrote {out} ({len(data)} bytes, sizes {sizes})')


if __name__ == '__main__':
    main()
