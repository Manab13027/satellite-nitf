"""
hires_sources.py - look for FREE high-resolution imagery over a point in three open catalogs
and package any hits in the Maxar delivery layout (via maxar_package.write_package).

  Maxar Open Data  (optical, ~0.3-0.5 m, CC BY-NC 4.0)  - disaster events only
  Capella Open Data (X-band SAR, ~0.5-1 m, CC BY 4.0)   - selected sites worldwide
  Umbra Open Data   (X-band SAR, ~0.25-1 m, CC BY 4.0)  - selected sites worldwide

The first run builds a local index of the Capella and Umbra catalogs (thousands of small JSON
files; a few minutes) and stores it in ~/.cache/get_nitf/. Later runs reuse it (refreshed
automatically after 30 days, or with --refresh-catalogs).
"""
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

from maxar_package import write_package, sun_position

CACHE = os.path.expanduser("~/.cache/get_nitf")
MAX_AGE_DAYS = 30
UA = {"User-Agent": "get_nitf/1.0"}

MAXAR_EVENTS = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
MAXAR_GEOJSON = "https://raw.githubusercontent.com/opengeos/maxar-open-data/master/datasets/{}.geojson"
CAPELLA_GEC = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
               "capella-open-data-by-product-type/capella-open-data-gec/collection.json")
UMBRA_BUCKET = "https://umbra-open-data-catalog.s3.amazonaws.com"
EARTH_R = 6371.0

LICENSES = {
    "maxar": ("Maxar Open Data Program imagery, licensed CC BY-NC 4.0 (non-commercial use only).\n"
              "Attribution: \"Imagery (c) Maxar Technologies (Vantor), Maxar Open Data Program, CC BY-NC 4.0\".\n"
              "https://registry.opendata.aws/maxar-open-data/  https://creativecommons.org/licenses/by-nc/4.0/\n"),
    "capella": ("Capella Space Open Data, licensed CC BY 4.0.\n"
                "Attribution: \"Capella Space Open Data, CC BY 4.0\".\n"
                "https://registry.opendata.aws/capella_opendata/  https://creativecommons.org/licenses/by/4.0/\n"),
    "umbra": ("Umbra Open Data Program, licensed CC BY 4.0.\n"
              "Attribution: \"Umbra Lab, Inc. Open Data Program, CC BY 4.0\".\n"
              "https://registry.opendata.aws/umbra-open-data/  https://creativecommons.org/licenses/by/4.0/\n"),
}


# ----------------------------------------------------------------------------- http + cache
def get(url, timeout=60, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def get_json(url):
    return json.loads(get(url))


def s3_to_https(href):
    if href.startswith("s3://"):
        b, _, key = href[5:].partition("/")
        return f"https://{b}.s3.amazonaws.com/{key}"
    return href


def cache_load(name, refresh):
    p = os.path.join(CACHE, name)
    if not refresh and os.path.exists(p) and time.time() - os.path.getmtime(p) < MAX_AGE_DAYS * 86400:
        with open(p) as f:
            return json.load(f)
    return None


def cache_save(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "w") as f:
        json.dump(obj, f)


def fetch_many(urls, fn, label, workers=32):
    out, done = [], 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, u): u for u in urls}
        for fu in cf.as_completed(futs):
            done += 1
            if done % 250 == 0 or done == len(urls):
                print(f"   indexing {label}: {done}/{len(urls)}", flush=True)
            try:
                r = fu.result()
                if r:
                    out.append(r)
            except Exception:
                pass
    return out


# ----------------------------------------------------------------------------- geometry
def find_geometry(obj):
    if isinstance(obj, dict):
        if obj.get("type") in ("Polygon", "MultiPolygon") and "coordinates" in obj:
            return obj
        for k in ("geometry", "footprintPolygonLla", "footprint"):
            if k in obj:
                g = find_geometry(obj[k])
                if g:
                    return g
        for v in obj.values():
            g = find_geometry(v)
            if g:
                return g
    elif isinstance(obj, list):
        for v in obj:
            g = find_geometry(v)
            if g:
                return g
    return None


def polys(g):
    return [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]


def bbox_of(g):
    xs, ys = [], []
    for p in polys(g):
        for x, y, *_ in p[0]:
            xs.append(x)
            ys.append(y)
    return [min(xs), min(ys), max(xs), max(ys)]


def in_bbox(b, lon, lat):
    return b[0] <= lon <= b[2] and b[1] <= lat <= b[3]


def in_geom(g, lon, lat):
    for p in polys(g):
        ring, c = p[0], False
        for i in range(len(ring)):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[i - 1][0], ring[i - 1][1]
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                c = not c
        if c:
            return True
    return False


def find_num(obj, pattern, exclude=None):
    """First numeric value whose (lower-case) key matches regex `pattern` (recursive)."""
    rx, ex = re.compile(pattern), re.compile(exclude) if exclude else None
    stack = [obj]
    while stack:
        o = stack.pop(0)
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and rx.search(kl) and not (ex and ex.search(kl)):
                    return float(v)
            stack += [v for v in o.values() if isinstance(v, (dict, list))]
        elif isinstance(o, list):
            stack += [v for v in o if isinstance(v, (dict, list))]
    return None


def find_str(obj, pattern):
    rx = re.compile(pattern)
    stack = [obj]
    while stack:
        o = stack.pop(0)
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and rx.search(k.lower()):
                    return v
            stack += [v for v in o.values() if isinstance(v, (dict, list))]
        elif isinstance(o, list):
            stack += [v for v in o if isinstance(v, (dict, list))]
    return None


def parse_time(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s[:26] + s[-6:] if "." in s and len(s) > 32 else s)
    except ValueError:
        t = dt.datetime.fromisoformat(s[:19])
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# ----------------------------------------------------------------------------- Maxar Open Data
def maxar_hits(lat, lon, refresh):
    events = cache_load("maxar_events.json", refresh)
    if events is None:
        cat = get_json(MAXAR_EVENTS)
        kids = [urllib.parse.urljoin(MAXAR_EVENTS, l["href"]) for l in cat["links"] if l.get("rel") == "child"]

        def one(u):
            c = get_json(u)
            bb = c.get("extent", {}).get("spatial", {}).get("bbox", [[-180, -90, 180, 90]])
            return {"id": c.get("id") or u.rstrip("/").split("/")[-2], "bboxes": bb}
        events = fetch_many(kids, one, "Maxar events", workers=16)
        cache_save("maxar_events.json", events)
    hits = []
    for ev in events:
        if not any(in_bbox(b, lon, lat) for b in ev["bboxes"]):
            continue
        name = f"maxar_{ev['id']}.geojson"
        fc = cache_load(name, refresh)
        if fc is None:
            try:
                fc = json.loads(get(MAXAR_GEOJSON.format(ev["id"]), timeout=300))
                cache_save(name, fc)
            except Exception as e:
                print(f"   (Maxar event {ev['id']}: footprints unavailable: {e})")
                continue
        seen = set()
        for f in fc.get("features", []):
            p = f["properties"]
            if not in_geom(f["geometry"], lon, lat) or p.get("catalog_id") in seen:
                continue
            seen.add(p.get("catalog_id"))
            off = p.get("view:off_nadir")
            plat = (p.get("platform") or "WV").upper()
            alt = 617.0 if plat == "WV03" else 770.0 if plat in ("WV02", "GE01") else 496.0
            inc = None
            if off is not None:
                inc = math.degrees(math.asin(min(1, (EARTH_R + alt) / EARTH_R * math.sin(math.radians(off)))))
            hits.append(dict(source="maxar", kind="optical", id=p.get("catalog_id"), event=ev["id"],
                             when=parse_time(p.get("datetime")), sat_id=plat, alt_km=alt,
                             href=s3_to_https(p.get("visual")), incidence=inc, off_nadir=off,
                             sat_az=p.get("view:azimuth"), sun_el=p.get("view:sun_elevation"),
                             sun_az=p.get("view:sun_azimuth"), gsd=p.get("gsd") or 0.5,
                             cloud=(p.get("tile:clouds_percent") or 0) / 100))
    return hits


# ----------------------------------------------------------------------------- Capella Open Data
def capella_index(refresh):
    idx = cache_load("capella_index.json", refresh)
    if idx is not None:
        return idx
    print("   building Capella index (one-time, a few minutes)...")
    col = get_json(CAPELLA_GEC)
    urls = [urllib.parse.urljoin(CAPELLA_GEC, l["href"]) for l in col["links"] if l.get("rel") == "item"]

    def one(u):
        it = get_json(u)
        g = find_geometry(it)
        return {"url": u, "bbox": it.get("bbox") or (bbox_of(g) if g else None)} if (g or it.get("bbox")) else None
    idx = fetch_many(urls, one, "Capella")
    cache_save("capella_index.json", idx)
    return idx


def stac_item_hit(source, url, lat, lon, alt_km):
    it = get_json(url)
    g = find_geometry(it)
    if g and not in_geom(g, lon, lat):
        return None
    props = it.get("properties", {})
    assets = {k: dict(v, href=s3_to_https(urllib.parse.urljoin(url, v["href"]))) for k, v in it.get("assets", {}).items()}
    img = None
    for k, v in assets.items():
        h = v["href"].lower()
        if h.endswith("_gec.tif") or (h.endswith((".tif", ".tiff")) and "thumb" not in k.lower()
                                       and "preview" not in k.lower() and "csi" not in h):
            img = v["href"]
            if h.endswith("_gec.tif"):
                break
    if not img:
        return None
    meta = props
    inc = find_num(props, r"incidence")
    graz = find_num(props, r"grazing")
    if inc is None and graz is None:  # try extended metadata asset
        for k, v in assets.items():
            if "metadata" in k.lower() or v["href"].lower().endswith("_extended.json"):
                try:
                    meta = get_json(v["href"])
                    inc = find_num(meta, r"incidence")
                    graz = find_num(meta, r"grazing")
                except Exception:
                    pass
                break
    if inc is None and graz is not None:
        inc = 90 - graz
    plat = (props.get("platform") or find_str(it, r"^platform$") or source).upper().replace("-", "")
    return dict(source=source, kind="sar", id=it.get("id"), when=parse_time(props.get("datetime") or
                props.get("start_datetime") or find_str(meta, r"start")), sat_id=plat[:12], alt_km=alt_km,
                href=img, incidence=inc,
                sat_az=find_num(props, r"(view:azimuth|satellite_azimuth|angleazimuth)") ,
                look=props.get("sar:observation_direction") or props.get("sat:orbit_state"),
                gsd=find_num(props, r"(gsd|pixel_spacing_range|resolution_range)") or 0.5, cloud=None)


def capella_hits(lat, lon, refresh):
    idx = capella_index(refresh)
    cand = [e["url"] for e in idx if e.get("bbox") and in_bbox(e["bbox"], lon, lat)]
    return [h for h in (stac_item_hit("capella", u, lat, lon, 525.0) for u in cand) if h]


# ----------------------------------------------------------------------------- Umbra Open Data
def s3_list(prefix):
    keys, token = [], None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(get(UMBRA_BUCKET + "/?" + urllib.parse.urlencode(q)))
        keys += [c.find("s:Key", ns).text for c in root.findall("s:Contents", ns)]
        if len(keys) % 20000 < 1000:
            print(f"   listing Umbra bucket: {len(keys)} files...", flush=True)
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return keys
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def umbra_index(refresh):
    idx = cache_load("umbra_index.json", refresh)
    if idx is not None:
        return idx
    print("   building Umbra index (one-time, a few minutes)...")
    keys = s3_list("sar-data/tasks/")
    by_dir = {}
    for k in keys:
        by_dir.setdefault(k.rsplit("/", 1)[0], []).append(k)
    jobs = []
    for d, ks in by_dir.items():
        meta = next((k for k in ks if k.endswith(".stac.v2.json")), None) or \
            next((k for k in ks if k.endswith("_METADATA.json")), None)
        gec = next((k for k in ks if k.endswith("_GEC.tif")), None)
        if meta and gec:
            jobs.append((f"{UMBRA_BUCKET}/{urllib.parse.quote(meta)}", f"{UMBRA_BUCKET}/{urllib.parse.quote(gec)}"))

    def one(job):
        m = get_json(job[0])
        g = find_geometry(m)
        return {"url": job[0], "gec": job[1], "bbox": bbox_of(g)} if g else None
    idx = fetch_many(jobs, one, "Umbra")
    cache_save("umbra_index.json", idx)
    return idx


def umbra_hits(lat, lon, refresh):
    out = []
    for e in umbra_index(refresh):
        if not (e.get("bbox") and in_bbox(e["bbox"], lon, lat)):
            continue
        m = get_json(e["url"])
        g = find_geometry(m)
        if g and not in_geom(g, lon, lat):
            continue
        inc = find_num(m, r"incidence")
        graz = find_num(m, r"grazing")
        if inc is None and graz is not None:
            inc = 90 - graz
        plat = (find_str(m, r"^(platform|satellite)") or "UMBRA").upper().replace("-", "")
        out.append(dict(source="umbra", kind="sar", id=m.get("id") or e["url"].rsplit("/", 1)[-1].split(".")[0],
                        when=parse_time(find_str(m, r"^(datetime|start_datetime|startatutc)")), sat_id=plat[:12],
                        alt_km=525.0, href=e["gec"], incidence=inc,
                        sat_az=find_num(m, r"(view:azimuth|angleazimuth|satellite_azimuth)"),
                        look=find_str(m, r"(observation_direction|lookside|side_of_track)"),
                        gsd=find_num(m, r"(gsd|resolution|pixel_spacing)", exclude=r"azimuth") or 0.5, cloud=None))
    return out


# ----------------------------------------------------------------------------- search + package
def find_hits(lat, lon, refresh=False, kinds=("optical", "sar")):
    hits = []
    jobs = []
    if "optical" in kinds:
        jobs.append(("Maxar Open Data", maxar_hits))
    if "sar" in kinds:
        jobs += [("Capella Open Data", capella_hits), ("Umbra Open Data", umbra_hits)]
    for label, fn in jobs:
        try:
            h = fn(lat, lon, refresh)
            print(f"   {label}: {len(h)} image(s) cover this point")
            hits += h
        except Exception as e:
            print(f"   {label}: could not be searched ({e})")
    return hits


def pick_diverse(hits, n):
    """Pick up to n hits with the most different (incidence, satellite azimuth)."""
    pool = [h for h in hits if h.get("href")]
    pool.sort(key=lambda h: (h.get("cloud") or 0, -(h["when"].timestamp() if h.get("when") else 0)))
    feat = lambda h: ((h.get("incidence") or 30) / 5, 2 * math.cos(math.radians(h.get("sat_az") or 0)),
                      2 * math.sin(math.radians(h.get("sat_az") or 0)))
    chosen = []
    while pool and len(chosen) < n:
        if not chosen:
            h = pool[0]
        else:
            h = max(pool, key=lambda c: min(sum((a - b) ** 2 for a, b in zip(feat(c), feat(o))) for o in chosen))
        pool.remove(h)
        chosen.append(h)
    return chosen + pool  # remaining kept as fallbacks, in order


def _read(href, lat, lon, half_m):
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        h = half_m
        if src.crs.is_geographic:
            h = half_m / 111320.0
            hx = h / max(math.cos(math.radians(lat)), 0.01)
        else:
            hx = h
        win = from_bounds(xs[0] - hx, ys[0] - h, xs[0] + hx, ys[0] + h, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(window=win, boundless=True, fill_value=0)
        if src.count > 3 and "visual" in href:
            data = data[:3]
        return data, src.window_transform(win), src.crs, src.nodata


def _stretch(a, valid, log=False):
    a = a.astype("float32")
    if log:
        a = np.where(a > 0, 20 * np.log10(np.maximum(a, 1e-6)), 0)
    v = a[valid]
    if v.size == 0:
        return np.zeros(a.shape, "uint8")
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    o = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    o[~valid] = 0
    return o.astype("uint8")


def package_hit(h, lat, lon, size_km, outdir):
    data, transform, crs, nod = _read(h["href"], lat, lon, size_km * 500)
    valid = (data != 0).any(axis=0) if nod is None else (data != nod).any(axis=0)
    if valid.mean() < 0.5:
        print(f"   skip {h['source']} {h['id']}: point is at the image edge ({valid.mean():.0%} data)")
        return None
    when = h["when"] or dt.datetime.now(dt.timezone.utc)
    sun_el, sun_az = h.get("sun_el"), h.get("sun_az")
    if sun_el is None:
        sun_el, sun_az = sun_position(when, lat, lon)
    inc = h.get("incidence")
    if inc is None:
        print(f"   note: {h['source']} {h['id']} has no incidence angle in its metadata; IMD angle fields = -999")
    gsd = abs(transform.a) if not crs.is_geographic else (h.get("gsd") or 0.5)
    if h["kind"] == "optical":
        rgb = data.shape[0] == 3 and data.dtype == np.uint8
        browse = np.transpose(data, (1, 2, 0)) if rgb else np.dstack([_stretch(data[min(i, data.shape[0] - 1)], valid) for i in (0, 1, 2)])
        band_names = ["R", "G", "B"][:data.shape[0]]
        band_id, band_code, icat, photo = "RGB", "S2AS", "VIS", ("RGB" if rgb else None)
    else:
        b = _stretch(data[0], valid, log=True)
        browse = np.dstack([b, b, b])
        band_names = [f"B{i + 1}" for i in range(data.shape[0])]
        band_id, band_code, icat, photo = "SAR", "R2AS", "SAR", None
    src_name = {"maxar": "Maxar Open Data", "capella": "Capella Open Data", "umbra": "Umbra Open Data"}[h["source"]]
    comments = [f"Source {src_name} {h['id']}", f"Satellite {h['sat_id']}",
                f"Incidence {inc:.1f} deg" if inc is not None else "Incidence unknown",
                f"Grazing {90 - inc:.1f} deg" if inc is not None else "", f"License {LICENSES[h['source']].splitlines()[0]}"]
    view = dict(incidence_deg=inc, sat_az_deg=h.get("sat_az") if h.get("sat_az") is not None else -999.0,
                sun_el_deg=sun_el, sun_az_deg=sun_az, in_track_deg=0.0)
    folder, base, extra = write_package(
        data=data, transform=transform, crs=crs, outroot=outdir, when=when, scene_id=str(h["id"]),
        sat_id=h["sat_id"], sensor_name=src_name, band_code=band_code, band_id=band_id,
        band_names=band_names, icat=icat, photometric=photo, gsd=round(gsd, 3), alt_km=h["alt_km"],
        view=view, cloud_frac=h.get("cloud"), browse_rgb=browse,
        license_text="LICENSE - SOURCE DATA\n=====================\n" + LICENSES[h["source"]],
        readme_extra=(f"Source catalog: {src_name}" + (f", event {h['event']}" if h.get("event") else "") +
                      f".\nOriginal image: {h['href']}\nCropped {size_km} km around {lat:.6f},{lon:.6f}; pixel values unchanged."
                      + ("" if inc is not None else "\nWARNING: incidence angle not found in source metadata; angle fields are placeholders.")),
        nitf_comments=[c for c in comments if c])
    graz = None if inc is None else round(90 - inc, 1)
    print(f"   saved {base}  [{src_name}] {when:%Y-%m-%d}  {h['sat_id']}  grazing {graz}°  ~{gsd:.2f} m")
    return {"sensor": src_name, "acquired_utc": when.isoformat(), "scene_id": h["id"], "platform": h["sat_id"],
            "grazing_angle_deg": graz, "incidence_deg_approx": None if inc is None else round(inc, 1),
            "view_azimuth_deg": h.get("sat_az"), "sun_elevation_deg": round(sun_el, 1),
            "off_nadir_deg": round(extra["off_nadir"], 2), "package": os.path.join(folder, base)}
