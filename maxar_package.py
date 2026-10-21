"""
maxar_package.py - write an image + its metadata as a Maxar/DigitalGlobe-style delivery folder:

    <BASE>.NTF          image (NITF 2.1, uncompressed)
    <BASE>.IMD          image metadata (DigitalGlobe IMD syntax, incl. sat/sun angles)
    <BASE>.RPB          RPC00B sensor model (fitted to the map-projected image)
    <BASE>.TIL          tile list
    <BASE>.XML          IMD + RPB + TIL combined as XML (<isd>)
    <BASE>_README.TXT   what the product is and where it came from
    <BASE>-BROWSE.JPG   browse/preview image
    LICENSE.TXT         licence of the source data

BASE follows the Maxar pattern  YYMONDDhhmmss-<band><level><type>-<orderid>_01_P001 .
The order id is a LOCAL id (starts with 9), not a Maxar order number. All satellite fields
name the real source satellite (Sentinel-1/-2). This is a format, not a Maxar product.
"""
import datetime as dt
import hashlib
import math
import os
from xml.sax.saxutils import escape

import numpy as np
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform as warp_transform

EARTH_R_KM = 6371.0


# ----------------------------------------------------------------------------- tiny ODL writer
class Raw(str):
    """Value written without quotes (dates, enums that DG writes bare)."""


class Group:
    def __init__(self, name, entries):
        self.name, self.entries = name, entries


def _fmt(v):
    if isinstance(v, Raw):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        t = f"{v:.9f}".rstrip("0")
        return t + "0" if t.endswith(".") else t
    raise TypeError(v)


def write_odl(path, entries):
    lines = []

    def emit(ents, ind):
        for k, v in ents:
            if isinstance(v, Group):
                lines.append(f"{ind}BEGIN_GROUP = {v.name}")
                emit(v.entries, ind + "\t")
                lines.append(f"{ind}END_GROUP = {v.name}")
            elif isinstance(v, (list, tuple)):
                body = (",\n" + ind + "\t\t\t").join(x if isinstance(x, str) else f"{x:+.15E}" for x in v)
                lines.append(f"{ind}{k} = (\n{ind}\t\t\t{body});")
            else:
                lines.append(f"{ind}{k} = {_fmt(v)};")

    emit(entries, "")
    lines.append("END;")
    with open(path, "w", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def odl_to_xml(entries, ind="\t"):
    out = []
    for k, v in entries:
        if isinstance(v, Group):
            out.append(f"{ind}<{v.name.upper()}>")
            out += odl_to_xml(v.entries, ind + "\t")
            out.append(f"{ind}</{v.name.upper()}>")
        elif isinstance(v, (list, tuple)):
            vals = " ".join(x if isinstance(x, str) else f"{x:+.15E}" for x in v)
            out.append(f"{ind}<{k.upper()}List><{k.upper()}>{escape(vals)}</{k.upper()}></{k.upper()}List>")
        else:
            s = str(v) if not isinstance(v, float) else _fmt(v)
            out.append(f"{ind}<{k.upper()}>{escape(s)}</{k.upper()}>")
    return out


# ----------------------------------------------------------------------------- geometry helpers
def sun_position(when, lat, lon):
    """Approximate solar elevation/azimuth (deg) - good to ~0.5 deg."""
    n = (when - dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)).total_seconds() / 86400
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    el = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(la) - math.sin(la) * math.cos(ha))
    return math.degrees(el), math.degrees(az) % 360


def off_nadir_from_incidence(inc_deg, alt_km):
    return math.degrees(math.asin(EARTH_R_KM / (EARTH_R_KM + alt_km) * math.sin(math.radians(inc_deg))))


def corners_lonlat(transform, crs, rows, cols):
    px = [(0, 0), (cols, 0), (cols, rows), (0, rows)]  # UL UR LR LL (outer pixel edges)
    xs = [transform.c + c * transform.a + r * transform.b for c, r in px]
    ys = [transform.f + c * transform.d + r * transform.e for c, r in px]
    lon, lat = warp_transform(crs, "EPSG:4326", xs, ys)
    return list(zip(lon, lat)), list(zip(xs, ys))


def fit_rpc(transform, crs, rows, cols, h_off=0.0, h_scale=1000.0, n=21):
    """Fit RPC00B (cubic, denominator = 1) mapping lat/lon/h -> line/sample for an ortho image.
    Line/sample 0,0 = centre of the first pixel."""
    r = np.linspace(0, rows - 1, n)
    c = np.linspace(0, cols - 1, n)
    R, C = np.meshgrid(r, c, indexing="ij")
    X = transform.c + (C + 0.5) * transform.a + (R + 0.5) * transform.b
    Y = transform.f + (C + 0.5) * transform.d + (R + 0.5) * transform.e
    lon, lat = warp_transform(crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist())
    lon, lat = np.array(lon), np.array(lat)
    lat_off, lon_off = float(lat.mean()), float(lon.mean())
    lat_sc = float(max(np.abs(lat - lat_off).max(), 1e-6))
    lon_sc = float(max(np.abs(lon - lon_off).max(), 1e-6))
    line_off, samp_off = (rows - 1) / 2, (cols - 1) / 2
    line_sc, samp_sc = max(rows / 2, 1), max(cols / 2, 1)
    P, L = (lat - lat_off) / lat_sc, (lon - lon_off) / lon_sc
    one = np.ones_like(P)
    # RPC00B term order: 1 L P H LP LH PH L2 P2 H2 PLH L3 LP2 LH2 L2P P3 PH2 L2H P2H H3 (H=0 here)
    terms = {0: one, 1: L, 2: P, 4: L * P, 7: L * L, 8: P * P, 11: L ** 3, 12: L * P * P, 14: L * L * P, 15: P ** 3}
    idx = sorted(terms)
    A = np.stack([terms[i] for i in idx], axis=1)
    yl = (R.ravel() - line_off) / line_sc
    ys = (C.ravel() - samp_off) / samp_sc
    cl, *_ = np.linalg.lstsq(A, yl, rcond=None)
    cs, *_ = np.linalg.lstsq(A, ys, rcond=None)
    line_num, samp_num = [0.0] * 20, [0.0] * 20
    for k, i in enumerate(idx):
        line_num[i], samp_num[i] = float(cl[k]), float(cs[k])
    den = [1.0] + [0.0] * 19
    rms = float(np.sqrt(np.mean(((A @ cl) * line_sc - (R.ravel() - line_off)) ** 2 +
                                ((A @ cs) * samp_sc - (C.ravel() - samp_off)) ** 2)))
    return dict(lineOffset=line_off, sampOffset=samp_off, latOffset=lat_off, longOffset=lon_off,
                heightOffset=h_off, lineScale=line_sc, sampScale=samp_sc, latScale=lat_sc,
                longScale=lon_sc, heightScale=h_scale, lineNumCoef=line_num, lineDenCoef=den,
                sampNumCoef=samp_num, sampDenCoef=den, rms_px=rms)


# ----------------------------------------------------------------------------- main entry
def make_base_name(when, band_code, scene_id):
    order = "9" + str(int(hashlib.sha1(scene_id.encode()).hexdigest(), 16) % 10 ** 11).zfill(11)
    stamp = when.strftime("%y") + when.strftime("%b").upper() + when.strftime("%d%H%M%S")
    return f"{stamp}-{band_code}-{order}_01_P001", order


def write_package(*, data, transform, crs, outroot, when, scene_id, sat_id, sensor_name, band_code,
                  band_id, band_names, icat, photometric, gsd, alt_km, view, cloud_frac,
                  browse_rgb, license_text, readme_extra, nitf_comments, abs_cal=None, bandwidths=None):
    """view: dict with incidence_deg, sat_az_deg, sun_el_deg, sun_az_deg, cross_track_deg, in_track_deg"""
    rows, cols = data.shape[1], data.shape[2]
    base, order = make_base_name(when, band_code, scene_id)
    folder = os.path.join(outroot, f"{order}_01")
    os.makedirs(folder, exist_ok=True)
    pfx = os.path.join(folder, base)
    ntf_name = base + ".NTF"

    # --- NTF
    prof = dict(driver="GTiff", width=cols, height=rows, count=data.shape[0], dtype=data.dtype,
                crs=crs, transform=transform)
    if photometric:
        prof["photometric"] = photometric
    with MemoryFile() as mem:
        with mem.open(**prof) as t:
            t.write(data)
        with mem.open() as t:
            rio_copy(t, pfx + ".NTF", driver="NITF", IC="NC", ICAT=icat,
                     FTITLE=f"{sensor_name} {scene_id}"[:80], IID1=band_code[:10], IID2=scene_id[:80],
                     ISORCE=sensor_name[:42], IDATIM=when.strftime("%Y%m%d%H%M%S"), ONAME="get_nitf.py",
                     ICOM=" | ".join(nitf_comments)[:720])

    # --- geometry
    (ul, ur, lr, ll), (mul, mur, mlr, mll) = corners_lonlat(transform, crs, rows, cols)
    inc = view["incidence_deg"]
    if inc is None:   # unknown geometry -> DG-style "no value"
        inc = off_nadir = sat_el = -999.0
    else:
        off_nadir = off_nadir_from_incidence(inc, alt_km)
        sat_el = 90.0 - inc
    epsg = crs.to_epsg() or 0
    zone, hemi = (epsg % 100, "N" if epsg // 100 == 326 else "S") if epsg // 100 in (326, 327) else (0, "N")
    now = dt.datetime.now(dt.timezone.utc)
    iso = lambda t: Raw(t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    corner_entries = lambda: [("ULLon", ul[0]), ("ULLat", ul[1]), ("ULHAE", 0.0),
                              ("URLon", ur[0]), ("URLat", ur[1]), ("URHAE", 0.0),
                              ("LRLon", lr[0]), ("LRLat", lr[1]), ("LRHAE", 0.0),
                              ("LLLon", ll[0]), ("LLLat", ll[1]), ("LLHAE", 0.0)]

    # --- IMD
    band_groups = []
    for i, bn in enumerate(band_names):
        ents = corner_entries()
        if abs_cal:
            ents += [("absCalFactor", abs_cal[i])]
        if bandwidths:
            ents += [("effectiveBandwidth", bandwidths[i])]
        band_groups.append(("", Group(f"BAND_{bn}", ents)))
    ang = lambda name, v: [(f"min{name}", v), (f"max{name}", v), (f"mean{name}", v)]
    image_grp = Group("IMAGE_1", [
        ("satId", sat_id), ("CatId", scene_id), ("SCANDIRECTION", "Forward"),
        ("firstLineTime", iso(when)),
        ("meanCollectedRowGSD", gsd), ("meanCollectedColGSD", gsd), ("meanCollectedGSD", gsd),
        ("meanProductRowGSD", gsd), ("meanProductColGSD", gsd), ("meanProductGSD", gsd),
        *ang("SunAz", round(view["sun_az_deg"], 1)), *ang("SunEl", round(view["sun_el_deg"], 1)),
        *ang("SatAz", round(view["sat_az_deg"], 1)), *ang("SatEl", round(sat_el, 1)),
        *ang("InTrackViewAngle", round(view.get("in_track_deg", 0.0), 1)),
        *ang("CrossTrackViewAngle", round(view.get("cross_track_deg", off_nadir) or off_nadir, 1)),
        *ang("OffNadirViewAngle", round(off_nadir, 1)),
        ("cloudCover", round(cloud_frac, 3) if cloud_frac is not None else -999.0),
        ("resamplingKernel", "CC"),
    ])
    map_grp = Group("MAP_PROJECTED_PRODUCT", [
        ("earliestAcqTime", iso(when)), ("latestAcqTime", iso(when)),
        ("datumName", "WE"), ("semiMajorAxis", 6378137.0), ("inverseFlattening", 298.257223563),
        ("datumOffset", ("0.000", "0.000", "0.000")),
        ("mapProjName", "UTM" if zone else "Unknown"), ("mapProjCode", 1 if zone else 0),
        ("mapZone", zone), ("mapHemi", hemi),
        ("productUnits", "M"), ("originX", transform.c), ("originY", transform.f),
        ("orientationAngle", 0.0), ("colSpacing", abs(transform.a)), ("rowSpacing", abs(transform.e)),
        ("productGSD", abs(transform.a)),
        ("ULX", mul[0]), ("ULY", mul[1]), ("ULH", 0.0), ("URX", mur[0]), ("URY", mur[1]), ("URH", 0.0),
        ("LRX", mlr[0]), ("LRY", mlr[1]), ("LRH", 0.0), ("LLX", mll[0]), ("LLY", mll[1]), ("LLH", 0.0),
        ("DEMCorrection", "Source-provided terrain correction"),
        ("terrainHAE", 0.0),
    ])
    imd = [("version", "28.4"), ("generationTime", iso(now)),
           ("productOrderId", f"{order}_01_P001"), ("productCatalogId", scene_id),
           ("imageDescriptor", "Standard2A"), ("bandId", band_id),
           ("panSharpenAlgorithm", "None"), ("numRows", rows), ("numColumns", cols),
           ("productLevel", "LV2A"), ("productType", "Standard"), ("numberOfLooks", 1),
           ("radiometricLevel", "Corrected"), ("radiometricEnhancement", "Off"),
           ("bitsPerPixel", int(np.dtype(data.dtype).itemsize * 8)), ("compressionType", "None"),
           *band_groups, ("outputFormat", "NITF21NCI"),
           ("", image_grp), ("", map_grp)]
    write_odl(pfx + ".IMD", imd)

    # --- RPB
    rpc = fit_rpc(transform, crs, rows, cols)
    rpb_img = Group("IMAGE", [("errBias", -1.0), ("errRand", -1.0)] +
                    [(k, rpc[k]) for k in ("lineOffset", "sampOffset", "latOffset", "longOffset",
                                           "heightOffset", "lineScale", "sampScale", "latScale",
                                           "longScale", "heightScale", "lineNumCoef", "lineDenCoef",
                                           "sampNumCoef", "sampDenCoef")])
    rpb = [("satId", sat_id), ("bandId", band_id), ("SpecId", "RPC00B"), ("", rpb_img)]
    write_odl(pfx + ".RPB", rpb)

    # --- TIL
    til = [("bandId", band_id), ("numTiles", 1), ("tileSizeX", cols), ("tileSizeY", rows),
           ("tileUnits", "Pixels"), ("tileOverlap", 0),
           ("", Group("TILE_1", [("filename", ntf_name), ("ULColOffset", 0), ("ULRowOffset", 0),
                                 ("URColOffset", cols - 1), ("URRowOffset", 0),
                                 ("LRColOffset", cols - 1), ("LRRowOffset", rows - 1),
                                 ("LLColOffset", 0), ("LLRowOffset", rows - 1),
                                 ("ULLon", ul[0]), ("ULLat", ul[1]), ("URLon", ur[0]), ("URLat", ur[1]),
                                 ("LRLon", lr[0]), ("LRLat", lr[1]), ("LLLon", ll[0]), ("LLLat", ll[1]),
                                 ("ULX", mul[0]), ("ULY", mul[1]), ("URX", mur[0]), ("URY", mur[1]),
                                 ("LRX", mlr[0]), ("LRY", mlr[1]), ("LLX", mll[0]), ("LLY", mll[1])]))]
    write_odl(pfx + ".TIL", til)

    # --- XML
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', "<isd>",
           "\t<IMD>", *odl_to_xml(imd, "\t\t"), "\t</IMD>",
           "\t<RPB>", *odl_to_xml(rpb, "\t\t"), "\t</RPB>",
           "\t<TIL>", *odl_to_xml(til, "\t\t"), "\t</TIL>", "</isd>"]
    with open(pfx + ".XML", "w", encoding="utf-8") as f:
        f.write("\n".join(xml) + "\n")

    # --- BROWSE.JPG
    try:
        from PIL import Image
        im = Image.fromarray(browse_rgb)
        im.thumbnail((1024, 1024))
        im.save(pfx + "-BROWSE.JPG", quality=85)
    except Exception as e:
        print(f"   (browse skipped: {e})")

    # --- README / LICENSE
    files = [ntf_name, base + ".IMD", base + ".RPB", base + ".TIL", base + ".XML",
             base + "_README.TXT", base + "-BROWSE.JPG", "LICENSE.TXT"]
    readme = f"""PRODUCT README
==============
Product            : {base}
Local order id     : {order}_01_P001   (generated locally - NOT a Maxar order number)
Source satellite   : {sat_id}  ({sensor_name})
Source scene id    : {scene_id}
Acquired (UTC)     : {when:%Y-%m-%d %H:%M:%S}
Created (UTC)      : {now:%Y-%m-%d %H:%M:%S}
Bands              : {band_id} ({', '.join(band_names)}), {np.dtype(data.dtype).name}
Size               : {cols} x {rows} pixels, {abs(transform.a):g} m, {crs}

Viewing geometry at scene centre
  Satellite elevation (grazing) : {sat_el:.1f} deg
  Incidence angle               : {inc:.1f} deg
  Off-nadir view angle          : {off_nadir:.1f} deg
  Satellite azimuth             : {view['sat_az_deg']:.1f} deg
  Sun elevation / azimuth       : {view['sun_el_deg']:.1f} / {view['sun_az_deg']:.1f} deg

{readme_extra}

This package uses the Maxar/DigitalGlobe delivery LAYOUT (NTF, IMD, RPB, TIL, XML, README,
BROWSE, LICENSE) so that tools which read that layout can open it. It is NOT a Maxar product
and contains no Maxar data.

The RPB is an RPC00B model fitted to the map-projected image (fit RMS {rpc['rms_px']:.4f} px). Because
the image is already terrain corrected, height terms are zero.

Files
-----
""" + "\n".join("  " + f for f in files) + "\n"
    with open(pfx + "_README.TXT", "w", newline="\r\n") as f:
        f.write(readme)
    with open(os.path.join(folder, "LICENSE.TXT"), "w", newline="\r\n") as f:
        f.write(license_text)
    return folder, base, dict(sat_el=sat_el, off_nadir=off_nadir, rpc_rms_px=rpc["rms_px"])


S2_LICENSE = """LICENSE - SOURCE DATA
=====================
Source: Copernicus Sentinel-2 Level-2A data (European Union / ESA), accessed via the
Sentinel-2 Cloud-Optimized GeoTIFF archive on AWS (Element 84 Earth Search).

Copernicus Sentinel data are provided free of charge under the Copernicus data policy
(full, free and open access) for any lawful use, commercial or non-commercial.
You may reproduce, distribute, adapt and modify the data.

Required attribution when you publish or share this data or products derived from it:
    "Contains modified Copernicus Sentinel data {year}"

The data are provided "as is" without warranty. See the full terms:
https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice
"""

S1_LICENSE = """LICENSE - SOURCE DATA
=====================
Source: Copernicus Sentinel-1 data (European Union / ESA), radiometrically terrain corrected
(RTC) product hosted on Microsoft Planetary Computer (collection "sentinel-1-rtc").

Copernicus Sentinel data are provided free of charge under the Copernicus data policy
(full, free and open access) for any lawful use, commercial or non-commercial.
The RTC processing layer is published by Microsoft Planetary Computer; check its licence
(CC BY 4.0 at the time of writing) at:
https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc

Required attribution when you publish or share this data or products derived from it:
    "Contains modified Copernicus Sentinel data {year}"
    plus credit to the RTC provider as stated on the dataset page.

The data are provided "as is" without warranty.
"""
