"""
images.py
BanglaIndoorVQA: THE one image loader (Design 4.9 and 5.2, decision D148).

Rule, in this order:
  1. apply the EXIF orientation;
  2. if the file carries a colour profile that is not sRGB, convert to sRGB;
  3. make it RGB;
  4. resize so the longer side is 1,024 px (Lanczos); a smaller image is not enlarged;
  5. save as JPEG quality 90, with no EXIF and no colour profile.

Every image is prepared ONCE into Drive data/images_1024/ and that one copy is
what all three models see. images_manifest.csv records the SHA-256 of every
prepared file. The harness refuses a prepared file whose bytes do not match.

Import this file. Do not write a second image loader anywhere.

Self check:  python src/images.py   (every line must print OK)
"""

import csv
import hashlib
import io
import os
from datetime import datetime, timezone

LONG_SIDE = 1024
JPEG_QUALITY = 90
PREP_VERSION = "img-v1: exif, srgb, rgb, lanczos 1024 no upscale, jpeg q90, no metadata"
MANIFEST_NAME = "images_manifest.csv"
MANIFEST_HEADER = ["image_file", "src_bytes", "src_width", "src_height", "exif_orientation",
                   "icc", "width", "height", "resized", "bytes", "sha256", "pillow",
                   "prepared_utc"]


class ImageError(Exception):
    """An image is missing, unreadable, or not the prepared copy."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _to_srgb(img, icc_bytes):
    """Convert to sRGB when a non-sRGB profile is embedded. Returns (img, note)."""
    if not icc_bytes:
        return img, "none"
    try:
        from PIL import ImageCms
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        name = (ImageCms.getProfileDescription(src) or "").strip().replace(",", " ")[:60]
        if "srgb" in name.lower():
            return img, "srgb"
        if img.mode not in ("RGB", "L", "CMYK"):
            img = img.convert("RGB")
        out = ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"),
                                        outputMode="RGB")
        return out, "converted:" + (name or "unnamed")
    except Exception as exc:  # an odd profile must not stop the run; it is counted
        return img, "failed:" + type(exc).__name__


def prepare_bytes(src_bytes):
    """Apply the rule to one image. Returns (jpeg_bytes, info dict)."""
    from PIL import Image, ImageOps
    import PIL
    try:
        img = Image.open(io.BytesIO(src_bytes))
        img.load()
    except Exception as exc:
        raise ImageError(f"cannot read the image ({type(exc).__name__}: {exc})")
    src_w, src_h = img.size
    orientation = img.getexif().get(0x0112, 1)
    icc_bytes = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)            # 1. EXIF orientation first
    img, icc_note = _to_srgb(img, icc_bytes)      # 2. colour profile
    if img.mode != "RGB":                         # 3. RGB
        img = img.convert("RGB")
    w, h = img.size
    resized = "no"
    if max(w, h) > LONG_SIDE:                     # 4. longer side to 1,024 px
        if w >= h:
            nw, nh = LONG_SIDE, max(1, round(h * LONG_SIDE / w))
        else:
            nw, nh = max(1, round(w * LONG_SIDE / h)), LONG_SIDE
        img = img.resize((nw, nh), Image.LANCZOS)
        resized = "yes"
    clean = Image.new("RGB", img.size)            # 5. a fresh image carries no metadata
    clean.paste(img)
    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    info = {"src_bytes": len(src_bytes), "src_width": src_w, "src_height": src_h,
            "exif_orientation": orientation, "icc": icc_note,
            "width": clean.size[0], "height": clean.size[1], "resized": resized,
            "bytes": len(data), "sha256": sha256_bytes(data), "pillow": PIL.__version__,
            "prepared_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return data, info


def manifest_path(dst_root):
    return os.path.join(str(dst_root), MANIFEST_NAME)


def read_manifest(dst_root):
    """Return {image_file: row}. A later row for the same file replaces an earlier one."""
    path = manifest_path(dst_root)
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("image_file"):
                rows[row["image_file"]] = row
    return rows


def _append_manifest(dst_root, row):
    path = manifest_path(dst_root)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(MANIFEST_HEADER)
        w.writerow([row[k] for k in MANIFEST_HEADER])
        f.flush()
        os.fsync(f.fileno())


def prepare_many(image_files, src_root, dst_root, say=print, verify=False):
    """Prepare every listed image that is not prepared yet. Can be run again at any time.

    image_files: relative paths as in the probe files, for example H01/H01_0001.jpg.
    Returns a dict of counts. Prints counts only.
    """
    os.makedirs(str(dst_root), exist_ok=True)
    manifest = read_manifest(dst_root)
    wanted = sorted(set(image_files))
    stats = {"wanted": len(wanted), "already": 0, "prepared": 0, "missing_source": 0,
             "unreadable": 0, "not_resized": 0, "icc_converted": 0, "icc_failed": 0,
             "rotated": 0, "verify_failed": 0}
    problems = []
    for n, rel in enumerate(wanted, 1):
        dst = os.path.join(str(dst_root), rel)
        row = manifest.get(rel)
        if row and os.path.exists(dst) and os.path.getsize(dst) == int(row["bytes"]):
            if verify:
                with open(dst, "rb") as f:
                    if sha256_bytes(f.read()) != row["sha256"]:
                        stats["verify_failed"] += 1
                        problems.append((rel, "prepared file differs from the manifest"))
                        continue
            stats["already"] += 1
            continue
        src = os.path.join(str(src_root), rel)
        if not os.path.exists(src):
            stats["missing_source"] += 1
            problems.append((rel, "source image not found"))
            continue
        try:
            with open(src, "rb") as f:
                data, info = prepare_bytes(f.read())
        except ImageError as exc:
            stats["unreadable"] += 1
            problems.append((rel, str(exc)))
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        info["image_file"] = rel
        _append_manifest(dst_root, info)
        manifest[rel] = {k: str(info[k]) for k in MANIFEST_HEADER}
        stats["prepared"] += 1
        stats["not_resized"] += 1 if info["resized"] == "no" else 0
        stats["icc_converted"] += 1 if info["icc"].startswith("converted") else 0
        stats["icc_failed"] += 1 if info["icc"].startswith("failed") else 0
        stats["rotated"] += 1 if info["exif_orientation"] not in (0, 1) else 0
        if n % 200 == 0:
            say(f"    {n} of {len(wanted)} checked, {stats['prepared']} prepared in this run")
    stats["problems"] = problems
    return stats


def load_prepared(dst_root, image_file, manifest):
    """Return (jpeg_bytes, sha256) of the prepared copy, checked against the manifest."""
    row = manifest.get(image_file)
    if row is None:
        raise ImageError(f"{image_file}: not in {MANIFEST_NAME}. Run prepare-images first.")
    path = os.path.join(str(dst_root), image_file)
    if not os.path.exists(path):
        raise ImageError(f"{image_file}: the prepared file is missing from {dst_root}.")
    with open(path, "rb") as f:
        data = f.read()
    digest = sha256_bytes(data)
    if digest != row["sha256"]:
        raise ImageError(f"{image_file}: the prepared file is not the one in the manifest. "
                         "Every model must see the same bytes (Design 5.2). Do not prepare "
                         "the images a second time; restore the file or ask in the chat.")
    return data, digest


def decode_rgb(jpeg_bytes):
    """Decode a prepared copy for the open models. No orientation step is needed."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg_bytes))
    img.load()
    return img if img.mode == "RGB" else img.convert("RGB")


if __name__ == "__main__":
    # Self check on made-up images. Every line must print OK.
    import tempfile
    from PIL import Image

    def make(w, h, exif_orientation=None, icc=None, mode="RGB"):
        img = Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else 128)
        if mode == "RGB":
            img.paste((30, 30, 200), (0, 0, w // 2, h // 4))   # a blue patch at top left
        kw = {}
        if exif_orientation:
            ex = Image.Exif()
            ex[0x0112] = exif_orientation
            kw["exif"] = ex.tobytes()
        if icc:
            kw["icc_profile"] = icc
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=95, **kw)
        return b.getvalue()

    checks = []
    data, info = prepare_bytes(make(4000, 3000))
    out = decode_rgb(data)
    checks.append(("4000x3000 becomes 1024x768", out.size == (1024, 768) and info["resized"] == "yes"))
    data, info = prepare_bytes(make(3000, 4000))
    checks.append(("3000x4000 becomes 768x1024", decode_rgb(data).size == (768, 1024)))
    data, info = prepare_bytes(make(800, 600))
    checks.append(("a small image is not enlarged", decode_rgb(data).size == (800, 600)
                   and info["resized"] == "no"))
    # Orientation 6: the stored pixels must be turned 90 degrees clockwise to be upright.
    data, info = prepare_bytes(make(4000, 3000, exif_orientation=6))
    out = decode_rgb(data)
    px = out.getpixel((out.size[0] - 20, 20))
    checks.append(("EXIF orientation 6 is applied before the resize",
                   out.size == (768, 1024) and info["exif_orientation"] == 6 and px[2] > px[0]))
    re_read = Image.open(io.BytesIO(data))
    checks.append(("the prepared copy has no EXIF and no colour profile",
                   len(re_read.getexif()) == 0 and "icc_profile" not in re_read.info))
    from PIL import ImageCms
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    data, info = prepare_bytes(make(2000, 1500, icc=srgb))
    checks.append(("an sRGB profile is recognised and left alone", info["icc"] == "srgb"))
    data, info = prepare_bytes(make(1200, 900, mode="L"))
    checks.append(("a grey image becomes RGB", decode_rgb(data).mode == "RGB"))
    a, _ = prepare_bytes(make(4000, 3000))
    b2, _ = prepare_bytes(make(4000, 3000))
    checks.append(("the same input gives the same bytes", a == b2))
    try:
        prepare_bytes(b"not an image")
        ok = False
    except ImageError:
        ok = True
    checks.append(("a broken file is refused", ok))
    tmp = tempfile.mkdtemp()
    src, dst = os.path.join(tmp, "images"), os.path.join(tmp, "images_1024")
    os.makedirs(os.path.join(src, "H01"))
    for name in ("H01_0001.jpg", "H01_0002.jpg"):
        with open(os.path.join(src, "H01", name), "wb") as f:
            f.write(make(4000, 3000))
    files = ["H01/H01_0001.jpg", "H01/H01_0002.jpg", "H01/H01_0003.jpg"]
    s1 = prepare_many(files, src, dst, say=lambda m: None)
    s2 = prepare_many(files, src, dst, say=lambda m: None, verify=True)
    checks.append(("prepare_many prepares 2 and reports 1 missing source",
                   s1["prepared"] == 2 and s1["missing_source"] == 1))
    checks.append(("a second run prepares nothing", s2["prepared"] == 0 and s2["already"] == 2))
    man = read_manifest(dst)
    got, digest = load_prepared(dst, files[0], man)
    checks.append(("load_prepared returns the manifest bytes", digest == man[files[0]]["sha256"]))
    with open(os.path.join(dst, files[1]), "ab") as f:
        f.write(b"x")
    try:
        load_prepared(dst, files[1], man)
        ok = False
    except ImageError:
        ok = True
    checks.append(("a changed prepared file is refused", ok))
    bad = 0
    for name, passed in checks:
        print(("OK   " if passed else "FAIL ") + name)
        bad += 0 if passed else 1
    print()
    print(f"{len(checks) - bad}/{len(checks)} self checks passed")
