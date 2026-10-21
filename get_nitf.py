#!/usr/bin/env python3
"""
get_nitf.py - Free satellite imagery around a lat/lon, delivered in the Maxar/DigitalGlobe
package layout (.NTF .IMD .RPB .TIL .XML _README.TXT -BROWSE.JPG LICENSE.TXT; see
maxar_package.py), with a choice of several viewing geometries. Keep maxar_package.py
in the same folder as this script, together with hires_sources.py.

By default it first checks three free HIGH-RESOLUTION catalogs (Maxar Open Data optical,
Capella and Umbra SAR) and uses any image covering the point; Sentinel fills the rest.

Sensors (both free, no account):
  optical : Sentinel-2 L2A, 10 m, true colour (or 4-band multispectral).
            Source: Element84 Earth Search (AWS). Near-nadir: 0-~11 deg off vertical.
  sar     : Sentinel-1 RTC (radiometrically terrain-corrected radar), 10 m, VV+VH in dB.
            Source: Microsoft Planetary Computer (anonymous token).
            Incidence ~29-46 deg  ->  grazing angle ~44-61 deg.

With --count N the script picks N scenes whose viewing geometry differs as much as
possible (different orbits / view azimuths / incidence and sun angles) and writes the
angles into each NITF (ICOM comments), each JSON, and a summary CSV.

Run (needs rasterio, numpy, pillow - already in ~/miniforge3):
    ~/miniforge3/bin/python get_nitf.py --lat 35.917777 --lon 74.328363
    ~/miniforge3/bin/python get_nitf.py --lat 35.917777 --lon 74.328363 --count 5 --sensor both
    ~/miniforge3/bin/python get_nitf.py --lat 35.917777 --lon 74.328363 --sensor sar --list
    ~/miniforge3/bin/python get_nitf.py --csv points.csv --count 5      # CSV: name,lat,lon

Attribution required: "Contains modified Copernicus Sentinel data <year>".
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maxar_package import write_package, sun_position, S1_LICENSE, S2_LICENSE  # noqa: E402
from hires_sources import find_hits, pick_diverse as pick_hires, package_hit  # noqa: E402

S2_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
S2_MS_BANDS = ["blue", "green", "red", "nir"]  # B02, B03, B04, B08 (10 m)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"
PC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/" + S1_COLLECTION

# Sentinel-1 IW swath incidence range (ellipsoid), degrees
S1_INC_NEAR, S1_INC_FAR = 29.1, 46.0

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_PAM_ENABLED", "NO")  # no .aux.xml side files
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


# ----------------------------------------------------------------------------- helpers
class Scene:
    def __init__(self, f):
        self.id = f["id"]
        self.properties = f["properties"]
        self.geometry = f.get("geometry")
        self.datetime = dt.datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00"))
        self.assets = {k: v["href"] for k, v in f["assets"].items()}
        self.angles = {}


def http_json(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stac_search(api, collection, lat, lon, start, end, extra=None, limit=300):
    body = {
        "collections": [collection],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": min(limit, 250),
    }
    if extra:
        body.update(extra)
    out, url = [], api + "/search"
    while url and len(out) < limit:
        res = http_json(url, body)
        out += [Scene(f) for f in res.get("features", [])]
        nxt = next((l for l in res.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url = nxt["href"]
        if nxt.get("method", "GET").upper() == "POST":
            nb = nxt.get("body") or {}
            body = {**body, **nb} if (nxt.get("merge") or "collections" not in nb) else nb
        else:
            body = None
        if body is None:  # GET pagination
            with urllib.request.urlopen(url, timeout=60) as r:
                res = json.load(r)
            out += [Scene(f) for f in res.get("features", [])]
            break
    return out


def read_crop(href, lat, lon, half_m, fill=0):
    """Read a square window (2*half_m metres wide) centred on lat/lon from a COG."""
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        win = from_bounds(x - half_m, y - half_m, x + half_m, y + half_m, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(window=win, boundless=True, fill_value=fill)
        return data, src.window_transform(win), src.crs, src.nodata


def stretch8(a, valid):
    v = a[valid]
    if v.size == 0:
        return np.zeros(a.shape, "uint8")
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    out = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    out[~valid] = 0
    return out.astype("uint8")


def save_png(rgb_hwc, path):
    try:
        from PIL import Image
        Image.fromarray(rgb_hwc).save(path)
    except Exception as e:
        print(f"   (preview skipped: {e})")


def write_nitf(data, transform, crs, path, icat, title, iid2, isorce, when, comments, photometric=None):
    profile = dict(driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                   dtype=data.dtype, crs=crs, transform=transform)
    if photometric:
        profile["photometric"] = photometric
    icom = " | ".join(comments)[:720]
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data)
        with mem.open() as tmp:
            rio_copy(tmp, path, driver="NITF", IC="NC", ICAT=icat,
                     FTITLE=title[:80], IID2=iid2[:80], ISORCE=isorce[:42],
                     IDATIM=when.strftime("%Y%m%d%H%M%S"), ONAME="get_nitf.py", ICOM=icom)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "point"


def circ_mean(deg_list):
    s = sum(math.sin(math.radians(d)) for d in deg_list)
    c = sum(math.cos(math.radians(d)) for d in deg_list)
    return round(math.degrees(math.atan2(s, c)), 3) % 360


def pick_diverse(cands, count, feat, first_key, try_fn):
    """Greedy max-min diversity selection. try_fn(scene) -> result or None (skip)."""
    remaining = sorted(cands, key=first_key)
    chosen, results = [], []
    while remaining and len(results) < count:
        if not chosen:
            s = remaining[0]
        else:
            def mind(c):
                fc = feat(c)
                return min(sum((a - b) ** 2 for a, b in zip(fc, feat(o))) for o in chosen)
            s = max(remaining, key=lambda c: (round(mind(c), 3), -first_key(c)[0] if isinstance(first_key(c), tuple) else 0))
        remaining.remove(s)
        r = try_fn(s)
        if r is not None:
            chosen.append(s)
            results.append(r)
    return results


# ----------------------------------------------------------------------------- Sentinel-2
def s2_rel_orbit(s):
    ro = s.properties.get("sat:relative_orbit")
    if ro is None:
        m = re.search(r"_R(\d{3})_", s.properties.get("s2:product_uri", ""))
        ro = int(m.group(1)) if m else None
    return ro


def s2_search(lat, lon, start, end, max_cloud):
    items = stac_search(S2_STAC, S2_COLLECTION, lat, lon, start, end,
                        extra={"query": {"eo:cloud_cover": {"lt": max_cloud}}})
    # one scene per acquisition (same pass can appear in 2 overlapping MGRS tiles)
    best = {}
    for s in items:
        k = (s.datetime.date(), s2_rel_orbit(s))
        if k not in best or s.properties.get("eo:cloud_cover", 100) < best[k].properties.get("eo:cloud_cover", 100):
            best[k] = s
    return list(best.values())


def s2_angles(s, lat, lon):
    """View zenith/azimuth and sun zenith/azimuth AT THE POINT from the granule metadata XML."""
    with urllib.request.urlopen(s.assets["granule_metadata"], timeout=60) as r:
        root = ET.fromstring(r.read())
    ln = lambda e: e.tag.rsplit("}", 1)[-1]
    epsg = ulx = uly = None
    for e in root.iter():
        t = ln(e)
        if t == "HORIZONTAL_CS_CODE":
            epsg = e.text.strip()
        elif t == "Geoposition" and e.get("resolution") == "10":
            for c in e:
                if ln(c) == "ULX":
                    ulx = float(c.text)
                elif ln(c) == "ULY":
                    uly = float(c.text)
    xs, ys = warp_transform("EPSG:4326", epsg, [lon], [lat])
    x, y = xs[0], ys[0]

    def grid_at(g):
        cs = rs = 5000.0
        vals = []
        for c in g:
            t = ln(c)
            if t == "COL_STEP":
                cs = float(c.text)
            elif t == "ROW_STEP":
                rs = float(c.text)
            elif t == "Values_List":
                vals = [[float(v) for v in row.text.split()] for row in c]
        rr, cc = (uly - y) / rs, (x - ulx) / cs
        got = []
        for r_ in {math.floor(rr), math.ceil(rr)}:
            for c_ in {math.floor(cc), math.ceil(cc)}:
                if 0 <= r_ < len(vals) and 0 <= c_ < len(vals[r_]) and not math.isnan(vals[r_][c_]):
                    got.append(vals[r_][c_])
        return got

    sun_z, sun_a, view_z, view_a = [], [], [], []
    mean_vz = mean_va = mean_sz = mean_sa = None
    for e in root.iter():
        t = ln(e)
        if t == "Sun_Angles_Grid":
            for c in e:
                if ln(c) == "Zenith":
                    sun_z += grid_at(c)
                elif ln(c) == "Azimuth":
                    sun_a += grid_at(c)
        elif t == "Viewing_Incidence_Angles_Grids" and e.get("bandId") == "3":  # B04 red
            for c in e:
                if ln(c) == "Zenith":
                    view_z += grid_at(c)
                elif ln(c) == "Azimuth":
                    view_a += grid_at(c)
        elif t == "Mean_Sun_Angle":
            for c in e:
                if ln(c) == "ZENITH_ANGLE":
                    mean_sz = float(c.text)
                elif ln(c) == "AZIMUTH_ANGLE":
                    mean_sa = float(c.text)
        elif t == "Mean_Viewing_Incidence_Angle" and e.get("bandId") == "3":
            for c in e:
                if ln(c) == "ZENITH_ANGLE":
                    mean_vz = float(c.text)
                elif ln(c) == "AZIMUTH_ANGLE":
                    mean_va = float(c.text)

    vz = sum(view_z) / len(view_z) if view_z else mean_vz
    va = circ_mean(view_a) if view_a else mean_va
    sz = sum(sun_z) / len(sun_z) if sun_z else mean_sz
    sa = circ_mean(sun_a) if sun_a else mean_sa
    return {
        "view_zenith_deg": round(vz, 2), "view_azimuth_deg": round(va, 1),
        "grazing_angle_deg": round(90 - vz, 2),
        "sun_elevation_deg": round(90 - sz, 2), "sun_azimuth_deg": round(sa, 1),
        "angles_at_point": bool(view_z),
    }


def s2_make(s, lat, lon, name, size_km, multispectral, outdir):
    half_m = size_km * 1000 / 2
    if multispectral:
        layers = []
        for b in S2_MS_BANDS:
            arr, transform, crs, _ = read_crop(s.assets[b], lat, lon, half_m)
            layers.append(arr[0])
        data = np.stack(layers)
        icat, bands = "MS", S2_MS_BANDS
    else:
        data, transform, crs, _ = read_crop(s.assets["visual"], lat, lon, half_m)
        icat, bands = "VIS", ["red", "green", "blue"]

    valid_frac = float((data.max(axis=0) > 0).mean())
    if valid_frac < 0.5:
        print(f"   skip {s.id}: only {valid_frac:.0%} of the crop has data (tile edge)")
        return None

    a = s.angles
    when = s.datetime
    if a.get("view_zenith_deg") is None:
        print(f"   skip {s.id}: no viewing angles available")
        return None
    plat = (s.properties.get("platform") or "sentinel-2").lower()
    sat_id = "S2" + plat[-1].upper() if plat[-1].isalpha() else "S2"
    if multispectral:
        valid = data[2] > 0
        browse = np.dstack([stretch8(data[i].astype("float32"), valid) for i in (2, 1, 0)])
        band_names, band_id, band_code = ["B", "G", "R", "N"], "Multi", "M2AS"
        abs_cal, bw = [1e-4] * 4, [0.066, 0.036, 0.031, 0.106]
    else:
        browse = np.transpose(data, (1, 2, 0))
        band_names, band_id, band_code = ["R", "G", "B"], "RGB", "C2AS"
        abs_cal, bw = None, None
    comments = [f"Source {sat_id} {s.id}", f"Relative orbit {s2_rel_orbit(s)}",
                f"View zenith {a['view_zenith_deg']} deg az {a['view_azimuth_deg']} deg",
                f"Satellite elevation {a['grazing_angle_deg']} deg",
                f"Sun elev {a['sun_elevation_deg']} az {a['sun_azimuth_deg']} deg",
                f"Contains modified Copernicus Sentinel data {when.year}"]
    folder, base, extra = write_package(
        data=data, transform=transform, crs=crs, outroot=outdir, when=when, scene_id=s.id,
        sat_id=sat_id, sensor_name="Copernicus Sentinel-2 MSI L2A", band_code=band_code,
        band_id=band_id, band_names=band_names, icat=icat, photometric=None if multispectral else "RGB",
        gsd=10.0, alt_km=786.0,
        view=dict(incidence_deg=a["view_zenith_deg"], sat_az_deg=a["view_azimuth_deg"],
                  sun_el_deg=a["sun_elevation_deg"], sun_az_deg=a["sun_azimuth_deg"], in_track_deg=0.0),
        cloud_frac=(s.properties.get("eo:cloud_cover") or 0) / 100, browse_rgb=browse,
        license_text=S2_LICENSE.format(year=when.year),
        readme_extra=("Processing: Sentinel-2 L2A surface reflectance, orthorectified by ESA. "
                      + ("16-bit, reflectance = DN x 1e-4 (check source offset)." if multispectral
                         else "8-bit true-colour (TCI) composite.")),
        nitf_comments=comments, abs_cal=abs_cal, bandwidths=bw)
    meta = {"sensor": "Sentinel-2 L2A", "name": name, "lat": lat, "lon": lon, "scene_id": s.id,
            "platform": sat_id, "relative_orbit": s2_rel_orbit(s), "acquired_utc": when.isoformat(),
            "cloud_cover_pct": s.properties.get("eo:cloud_cover"), **a,
            "off_nadir_deg": round(extra["off_nadir"], 2), "package": os.path.join(folder, base)}
    print(f"   saved {base}  {when:%Y-%m-%d}  sat elev {a['grazing_angle_deg']}°  sat az {a['view_azimuth_deg']}°"
          f"  sun elev {a['sun_elevation_deg']}°")
    return meta


def run_optical(lat, lon, name, a, outdir):
    items = s2_search(lat, lon, a.start, a.end, a.max_cloud)
    print(f"[{name}] Sentinel-2: {len(items)} acquisitions with cloud < {a.max_cloud}%")
    if not items:
        return []
    if a.count == 1 and a.pick == "latest":
        items = [max(items, key=lambda s: s.datetime)]
    items.sort(key=lambda s: (round(s.properties.get("eo:cloud_cover", 100), 1), -s.datetime.timestamp()))
    cands = items[: max(25, a.count * 6)] if a.count > 1 else items[:3]
    ok = []
    for s in cands:
        try:
            s.angles = s2_angles(s, lat, lon)
            ok.append(s)
        except Exception as e:
            print(f"   (angles unavailable for {s.id}: {e})")
            if a.count == 1:
                ok.append(s)
    feat = lambda s: (s.angles.get("view_zenith_deg", 0) / 2,
                      2 * math.cos(math.radians(s.angles.get("view_azimuth_deg", 0))),
                      2 * math.sin(math.radians(s.angles.get("view_azimuth_deg", 0))),
                      s.angles.get("sun_elevation_deg", 45) / 10)
    first = lambda s: (round(s.properties.get("eo:cloud_cover", 100), 1), -s.datetime.timestamp())

    def try_fn(s):
        try:
            return s2_make(s, lat, lon, name, a.size_km, a.multispectral, outdir)
        except Exception as e:
            print(f"   skip {s.id}: {e}")
            return None
    return pick_diverse(ok, a.count, feat, first, try_fn)


# ----------------------------------------------------------------------------- Sentinel-1
def s1_token():
    return http_json(PC_TOKEN_URL)["token"]


def s1_incidence(s, lat, lon):
    """Approximate ellipsoid incidence at the point from its across-track position in the swath."""
    g = s.geometry or {}
    polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", [])
    xs = []
    for poly in polys:
        ring = poly[0]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            if (y1 - lat) * (y2 - lat) <= 0 and y1 != y2:
                xs.append(x1 + (lat - y1) * (x2 - x1) / (y2 - y1))
    if len(xs) < 2:
        return None
    w, e = min(xs), max(xs)
    f = min(max((lon - w) / max(e - w, 1e-9), 0), 1)
    asc = s.properties.get("sat:orbit_state", "").lower().startswith("asc")
    frac_from_near = f if asc else 1 - f  # right-looking: ascending looks east, descending looks west
    inc = S1_INC_NEAR + frac_from_near * (S1_INC_FAR - S1_INC_NEAR)
    return {"incidence_deg_approx": round(inc, 1), "grazing_angle_deg_approx": round(90 - inc, 1),
            "look_direction": "east (ENE)" if asc else "west (WNW)",
            "orbit_state": s.properties.get("sat:orbit_state"),
            "relative_orbit": s.properties.get("sat:relative_orbit")}


def s1_search(lat, lon, start, end):
    items = stac_search(PC_STAC, S1_COLLECTION, lat, lon, start, end, limit=400)
    for s in items:
        s.angles = s1_incidence(s, lat, lon) or {}
    return items


def s1_make(s, token, lat, lon, name, size_km, outdir):
    half_m = size_km * 1000 / 2
    pols = [p for p in ("vv", "vh") if p in s.assets]
    layers, transform, crs = [], None, None
    for p in pols:
        arr, transform, crs, nod = read_crop(s.assets[p] + "?" + token, lat, lon, half_m, fill=0)
        lin = arr[0].astype("float32")
        good = np.isfinite(lin) & (lin > 0)
        if nod is not None:
            good &= lin != nod
        db = np.full(lin.shape, -99.0, "float32")
        db[good] = 10 * np.log10(lin[good])
        layers.append(db)
    data = np.stack(layers)
    valid = data[0] > -99
    if valid.mean() < 0.5:
        print(f"   skip {s.id}: only {valid.mean():.0%} of the crop has data")
        return None

    a, when = s.angles, s.datetime
    if a.get("incidence_deg_approx") is None:
        print(f"   skip {s.id}: incidence unknown")
        return None
    plat = (s.properties.get("platform") or "sentinel-1").lower()
    sat_id = "S1" + plat[-1].upper() if plat[-1].isalpha() else "S1"
    asc = (a.get("orbit_state") or "").lower().startswith("asc")
    ci = math.cos(math.radians(98.18)) / max(math.cos(math.radians(lat)), 1e-3)
    drift = math.degrees(math.asin(max(-1, min(1, ci))))   # ~ -10 deg (sun-synchronous, retrograde)
    heading = (360 + drift) % 360 if asc else (180 - drift) % 360   # asc ~350 (NNW), desc ~190 (SSW)
    look_az = (heading + 90) % 360          # right-looking
    sat_az = (look_az + 180) % 360          # direction from ground to satellite
    sun_el, sun_az = sun_position(when, lat, lon)
    vv = stretch8(data[0], valid)
    browse = np.dstack([vv, stretch8(data[1], valid), stretch8(data[0] - data[1], valid)]) if len(pols) == 2 \
        else np.dstack([vv, vv, vv])
    comments = [f"Source {sat_id} {s.id}", f"Orbit {a.get('orbit_state')} rel {a.get('relative_orbit')}",
                f"Incidence approx {a['incidence_deg_approx']} deg (ellipsoid)",
                f"Grazing approx {a['grazing_angle_deg_approx']} deg", f"Look azimuth {look_az:.0f} deg",
                f"{','.join(p.upper() for p in pols)} gamma0 RTC dB, nodata -99",
                f"Contains modified Copernicus Sentinel data {when.year}"]
    folder, base, extra = write_package(
        data=data, transform=transform, crs=crs, outroot=outdir, when=when, scene_id=s.id,
        sat_id=sat_id, sensor_name="Copernicus Sentinel-1 C-SAR IW RTC", band_code="R2AS",
        band_id="_".join(p.upper() for p in pols), band_names=[p.upper() for p in pols], icat="SAR",
        photometric=None, gsd=10.0, alt_km=693.0,
        view=dict(incidence_deg=a["incidence_deg_approx"], sat_az_deg=sat_az, sun_el_deg=sun_el,
                  sun_az_deg=sun_az, in_track_deg=0.0),
        cloud_frac=None, browse_rgb=browse, license_text=S1_LICENSE.format(year=when.year),
        readme_extra=("Processing: Sentinel-1 IW GRD, radiometrically terrain corrected (gamma0), "
                      "converted to dB (float32, nodata -99). Browse: R=VV G=VH B=VV-VH.\n"
                      f"Look direction {a.get('look_direction')} (azimuth {look_az:.0f} deg). Incidence/grazing "
                      "are ellipsoid approximations from the swath position; local terrain slope changes "
                      "the true local angle. Sun angles are computed and only informative for radar."),
        nitf_comments=comments)
    meta = {"sensor": "Sentinel-1 RTC (IW)", "name": name, "lat": lat, "lon": lon, "scene_id": s.id,
            "platform": sat_id, "acquired_utc": when.isoformat(), **a,
            "off_nadir_deg": round(extra["off_nadir"], 2), "package": os.path.join(folder, base)}
    print(f"   saved {base}  {when:%Y-%m-%d}  {a.get('orbit_state')} orbit {a.get('relative_orbit')}"
          f"  incidence ~{a['incidence_deg_approx']}°  grazing ~{a['grazing_angle_deg_approx']}°")
    return meta


def run_sar(lat, lon, name, a, outdir):
    items = s1_search(lat, lon, a.start, a.end)
    print(f"[{name}] Sentinel-1: {len(items)} acquisitions")
    if not items:
        return []
    groups = {}
    for s in items:
        groups.setdefault((s.angles.get("orbit_state"), s.angles.get("relative_orbit")), []).append(s)
    print(f"   {len(groups)} distinct orbit tracks over this point:")
    for (st, ro), ss in sorted(groups.items(), key=lambda kv: kv[1][0].angles.get("incidence_deg_approx") or 0):
        print(f"     {st} orbit {ro}: incidence ~{ss[0].angles.get('incidence_deg_approx')}°, {len(ss)} dates")
    for ss in groups.values():
        ss.sort(key=lambda s: s.datetime, reverse=True)
    # round-robin across tracks (most distinct geometry first), newest date per track
    order, depth = [], 0
    while len(order) < len(items):
        added = False
        for ss in groups.values():
            if depth < len(ss):
                order.append(ss[depth])
                added = True
        if not added:
            break
        depth += 1
    if a.count == 1 and a.pick == "latest":
        order = sorted(items, key=lambda s: s.datetime, reverse=True)
    token = s1_token()
    results = []
    for s in order:
        if len(results) >= a.count:
            break
        try:
            r = s1_make(s, token, lat, lon, name, a.size_km, outdir)
        except Exception as e:
            print(f"   skip {s.id}: {e}")
            r = None
        if r:
            results.append(r)
    if len(groups) < a.count:
        print(f"   note: only {len(groups)} distinct tracks exist here, so some images share a viewing angle "
              f"(different dates). Widen --start/--end for more variety in dates, not angles.")
    return results


# ----------------------------------------------------------------------------- main
def write_summary(rows, path):
    if not rows:
        return
    cols = ["sensor", "acquired_utc", "scene_id", "platform", "relative_orbit", "orbit_state",
            "view_zenith_deg", "view_azimuth_deg", "grazing_angle_deg", "incidence_deg_approx",
            "grazing_angle_deg_approx", "look_direction", "sun_elevation_deg", "sun_azimuth_deg",
            "cloud_cover_pct", "off_nadir_deg", "package"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"   summary: {path}")


def main():
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Free Sentinel-2 / Sentinel-1 imagery -> NITF for a lat/lon.")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--name", default=None, help="label used in file names")
    p.add_argument("--csv", help="CSV with columns name,lat,lon for several points")
    p.add_argument("--sensor", choices=["optical", "sar", "both"], default="both")
    p.add_argument("--count", type=int, default=5, help="images per sensor, chosen for varied viewing angles (default 5)")
    p.add_argument("--size-km", type=float, default=5.0, help="width of square crop (default 5 km)")
    p.add_argument("--start", default=str(today - dt.timedelta(days=365)), help="YYYY-MM-DD (default: 1 year ago)")
    p.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    p.add_argument("--max-cloud", type=float, default=10, help="optical: max scene cloud %% (default 10)")
    p.add_argument("--pick", choices=["clearest", "latest"], default="clearest", help="for --count 1")
    p.add_argument("--multispectral", action="store_true", help="optical: 4-band 16-bit B,G,R,NIR")
    p.add_argument("--list", action="store_true", help="only list available scenes")
    p.add_argument("--out", default="output", help="output folder (default ./output)")
    p.add_argument("--no-hires", action="store_true",
                   help="skip the free high-res catalogs (Maxar Open Data, Capella, Umbra)")
    p.add_argument("--hires-size-km", type=float, default=1.0, help="crop width for high-res images (default 1 km)")
    p.add_argument("--refresh-catalogs", action="store_true", help="rebuild the cached high-res catalog index")
    a = p.parse_args()

    points = []
    if a.csv:
        with open(a.csv, newline="") as f:
            for row in csv.DictReader(f):
                points.append((row.get("name") or f"{row['lat']}_{row['lon']}", float(row["lat"]), float(row["lon"])))
    elif a.lat is not None and a.lon is not None:
        points.append((a.name or f"{a.lat:.5f}_{a.lon:.5f}", a.lat, a.lon))
    else:
        p.error("give --lat and --lon, or --csv")

    for name, lat, lon in points:
        outdir = os.path.join(a.out, safe_name(name))
        if a.list:
            if a.sensor in ("optical", "both"):
                items = sorted(s2_search(lat, lon, a.start, a.end, a.max_cloud), key=lambda s: s.datetime)
                print(f"[{name}] Sentinel-2: {len(items)} scenes")
                for s in items:
                    print(f"   {s.datetime:%Y-%m-%d}  cloud={s.properties.get('eo:cloud_cover', 0):5.1f}%"
                          f"  orbit R{s2_rel_orbit(s)}  {s.id}")
            if a.sensor in ("sar", "both"):
                items = sorted(s1_search(lat, lon, a.start, a.end), key=lambda s: s.datetime)
                print(f"[{name}] Sentinel-1: {len(items)} scenes")
                for s in items:
                    g = s.angles
                    print(f"   {s.datetime:%Y-%m-%d}  {g.get('orbit_state')} orbit {g.get('relative_orbit')}"
                          f"  incidence ~{g.get('incidence_deg_approx')}°  {s.id}")
            continue

        os.makedirs(outdir, exist_ok=True)
        rows = []
        kinds = [k for k in ("optical", "sar") if a.sensor in (k, "both")]
        made = {k: 0 for k in kinds}
        if not a.no_hires:
            print(f"[{name}] Checking free high-resolution catalogs...")
            try:
                hits = find_hits(lat, lon, refresh=a.refresh_catalogs, kinds=kinds)
            except Exception as e:
                print(f"   high-res search failed: {e}")
                hits = []
            for k in kinds:
                for h in pick_hires([h for h in hits if h["kind"] == k], len(hits)):
                    if made[k] >= a.count:
                        break
                    try:
                        r = package_hit(h, lat, lon, a.hires_size_km, outdir)
                    except Exception as e:
                        print(f"   skip {h['source']} {h['id']}: {e}")
                        r = None
                    if r:
                        rows.append(r)
                        made[k] += 1
        for sensor, fn in (("optical", run_optical), ("sar", run_sar)):
            if sensor in kinds and made[sensor] < a.count:
                sub = argparse.Namespace(**vars(a))
                sub.count = a.count - made[sensor]
                if made[sensor]:
                    print(f"[{name}] {made[sensor]} high-res {sensor} image(s) found; adding {sub.count} from Sentinel")
                try:
                    rows += [dict(r) for r in fn(lat, lon, name, sub, outdir)]
                except Exception as e:
                    print(f"[{name}] {sensor} ERROR: {e}", file=sys.stderr)
        write_summary(rows, os.path.join(outdir, f"{safe_name(name)}_angles.csv"))
        print(f"[{name}] done: {len(rows)} NITF files in {outdir}")


if __name__ == "__main__":
    main()
