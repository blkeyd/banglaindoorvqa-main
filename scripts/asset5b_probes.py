"""
asset5b_probes.py
BanglaIndoorVQA, ASSET-5B: draw P-NAME (1,000 items) and P-CLUST.

Rules (D42, D57, D58, D59; clusters per D41 and D43):
  Eligible   close-up obj_name items whose gold name has at least 10 obj_name
             answers, or at least 5 if the name is culturally_specific in the
             frozen ASSET-4A file. Counted per normalise()d name (D57).
  Protected  every eligible culturally_specific item, and every item whose
             gold name is a member of a qualifying cluster, whatever its count
             (D58). Protected items are all taken.
  Fill       the rest of the 1,000 from the other eligible items, lowest
             SHA-256("42|P-NAME|<qa_id>") first.
  P-CLUST    the P-NAME items in qualifying clusters. Because of D58 this is
             every obj_name item of every qualifying-cluster member.
  E4-600     the 600 P-NAME items with the lowest SHA-256("42|E4|<qa_id>"),
             flagged in_E4_600 (D59).
  Clusters   ASSET-1's clusters_parsed.csv with D43 applied: ডাস্টবিন leaves
             অন্যান্য জোড়া, which becomes টব / ফুলদানি, and বদনা leaves
             পানি ও পরিবেশনের পাত্র so it sits only in the bathroom cluster.
             A cluster qualifies with 30 or more obj_name items and at least
             two members at 8 or more; this reproduces the G1 reading.

Nothing is written unless all of these hold:
  every member's recount equals ASSET-1's cluster_counts.csv;
  the qualifying clusters match the G1 reading (7 clusters, totals, members);
  no surface form belongs to two clusters;
  P-NAME has exactly 1,000 unique items and holds every protected item.

Subcommands:
  selftest  end to end on SYNTHETIC data; every check must fire when broken
  draw      the real draw; writes P-NAME.json, P-CLUST.json, probe_manifest_5B.json

Needs normalise.py, asset5a_probes.py and asset4a_specificity.py in the same
folder. Never copy normalise() into this file.
"""

import argparse
import contextlib
import datetime
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

from normalise import normalise
import asset5a_probes as a5
import asset4a_specificity as a4

say = a5.say
hkey = a5.hkey
sha256_file = a5.sha256_file

SEED = 42
PNAME_SIZE = 1000
E4_SIZE = 600
FLOOR_CS = 5         # D42
FLOOR_OTHER = 10     # Guide 3.2, kept by D42; counted on obj_name answers (D57)
QUAL_MIN_TOTAL = 30
QUAL_MIN_MEMBER = 8
QUAL_MIN_MEMBERS = 2
CS = "culturally_specific"
# G1 reading, 2026-09-10 (PROJECT_STATE.md): (obj_name total, members at 8+) of
# the seven qualifying clusters.
G1_READING = sorted([(138, 5), (88, 4), (79, 5), (61, 4), (43, 2), (42, 2), (41, 3)], reverse=True)
# D43, applied by name so the edit is visible and checked.
D43_DROP = [("অন্যান্য জোড়া", "ডাস্টবিন"), ("পানি ও পরিবেশনের পাত্র", "বদনা")]
D43_RENAME = {"অন্যান্য জোড়া": "টব / ফুলদানি"}

ITEM_KEYS = ["qa_id", "image_id", "house_id", "capture_type", "qa_type", "question_bn",
             "answer_bn", "image_file", "name_bn", "name_id", "cultural_specificity",
             "cluster_id", "cluster_member", "in_qualifying_cluster", "protected", "in_E4_600"]


# ----------------------------------------------------------------------------
# Clusters
# ----------------------------------------------------------------------------
def _find(clusters, cid):
    return next((k for k in clusters if normalise(k) == normalise(cid)), None)


def load_clusters(path):
    """cluster id -> list of members {member, forms, orig_cluster}, with D43 applied."""
    rows = a5.read_csv(path)
    need = {"cluster_id", "member", "surface_forms"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"[clusters] ERROR: {path} needs columns {sorted(need)}; it has "
                         f"{list(rows[0]) if rows else 'no rows'}")
    clusters = {}
    for r in rows:
        forms = [normalise(x) for x in (r["surface_forms"] or "").split("|") if normalise(x)]
        if not forms:
            forms = [normalise(r["member"])]
        cid = r["cluster_id"].strip()
        clusters.setdefault(cid, []).append({"member": r["member"].strip(), "forms": forms,
                                             "orig_cluster": cid})
    say(f"[clusters] {len(rows)} members in {len(clusters)} clusters from {path}")
    for cid, form in D43_DROP:
        key = _find(clusters, cid)
        if key is None:
            raise SystemExit(f"[clusters] ERROR: D43 names cluster '{cid}', which is not in {path}")
        before = len(clusters[key])
        clusters[key] = [m for m in clusters[key] if normalise(form) not in m["forms"]]
        if len(clusters[key]) == before:
            raise SystemExit(f"[clusters] ERROR: D43 removes '{form}' from '{cid}', but it is not there")
        say(f"[clusters] D43: removed {form} from {key}")
    for old, new in D43_RENAME.items():
        key = _find(clusters, old)
        if key is None:
            raise SystemExit(f"[clusters] ERROR: D43 renames '{old}', which is not in {path}")
        clusters[new] = clusters.pop(key)
        say(f"[clusters] D43: {key} is now {new}")
    owner = {}
    for cid, ms in clusters.items():
        for m in ms:
            for f in m["forms"]:
                if f in owner and owner[f][0] != cid:
                    raise SystemExit(f"[clusters] ERROR: surface form '{f}' is in two clusters, "
                                     f"'{owner[f][0]}' and '{cid}'. A term in two clusters "
                                     f"double counts (D43). Stop and ask.")
                owner[f] = (cid, m["member"])
    return clusters, owner


def recount(clusters, answer_counts):
    counts = {}
    for cid, ms in clusters.items():
        for m in ms:
            counts[(cid, m["member"])] = sum(answer_counts.get(f, 0) for f in m["forms"])
    return counts


def check_against_asset1(clusters, counts, cc_path):
    rows = a5.read_csv(cc_path)
    if not rows or not {"cluster_id", "member", "obj_name_count"} <= set(rows[0]):
        raise SystemExit(f"[check] ERROR: {cc_path} needs cluster_id, member, obj_name_count")
    expected = {(r["cluster_id"].strip(), r["member"].strip()): int(r["obj_name_count"]) for r in rows}
    diffs, compared = [], 0
    for cid, ms in clusters.items():
        for m in ms:
            key = (m["orig_cluster"], m["member"])
            if key not in expected:
                diffs.append((cid, m["member"], "not in cluster_counts.csv", counts[(cid, m["member"])]))
                continue
            compared += 1
            if expected[key] != counts[(cid, m["member"])]:
                diffs.append((cid, m["member"], expected[key], counts[(cid, m["member"])]))
    if diffs:
        raise SystemExit(f"[check] ERROR: {len(diffs)} cluster members recount differently from "
                         f"ASSET-1's {os.path.basename(cc_path)}, e.g. {diffs[:3]} "
                         f"(cluster, member, ASSET-1, now). Stop and ask.")
    say(f"[check] all {compared} cluster members recount exactly as in {os.path.basename(cc_path)}")


def qualifying(clusters, counts):
    out = {}
    for cid, ms in clusters.items():
        total = sum(counts[(cid, m["member"])] for m in ms)
        ge8 = sum(1 for m in ms if counts[(cid, m["member"])] >= QUAL_MIN_MEMBER)
        if total >= QUAL_MIN_TOTAL and ge8 >= QUAL_MIN_MEMBERS:
            out[cid] = (total, ge8)
    got = sorted(out.values(), reverse=True)
    if got != G1_READING:
        raise SystemExit(f"[check] ERROR: qualifying clusters {got} (total, members at 8+) do "
                         f"not match the G1 reading {G1_READING}. Stop and ask.")
    say(f"[check] the {len(out)} qualifying clusters match the G1 reading exactly")
    return out


# ----------------------------------------------------------------------------
# Draw
# ----------------------------------------------------------------------------
def draw_pname(items, tags, clusters, owner, qual):
    obj = [it for it in items if it["qa_type"] == "obj_name"]
    n_by_name = Counter(normalise(it["answer_bn"]) for it in obj)
    rows = []
    for it in obj:
        n = normalise(it["answer_bn"])
        nid = a4.name_id(n)
        tag = tags.get(nid, "")
        cid, member = owner.get(n, ("", ""))
        inq = cid in qual
        floor = FLOOR_CS if tag == CS else FLOOR_OTHER
        eligible = n_by_name[n] >= floor
        protected = "culturally_specific" if (tag == CS and eligible) else ""
        if inq:
            protected = "both" if protected else "cluster"
        rows.append(dict(it, name_bn=n, name_id=nid, cultural_specificity=tag, cluster_id=cid,
                         cluster_member=member, in_qualifying_cluster=inq,
                         protected=protected, _eligible=eligible))
    prot = [r for r in rows if r["protected"]]
    rest = sorted((r for r in rows if not r["protected"] and r["_eligible"]),
                  key=lambda r: hkey(SEED, "P-NAME", r["qa_id"]))
    need = PNAME_SIZE - len(prot)
    say(f"[draw] eligible or protected pool: {len(prot) + len(rest)} items; protected {len(prot)}; "
        f"filling {need} from {len(rest)} others")
    if need < 0:
        raise SystemExit(f"[draw] ERROR: {len(prot)} protected items exceed {PNAME_SIZE}. Stop and ask.")
    if len(rest) < need:
        raise SystemExit(f"[draw] ERROR: only {len(rest)} other eligible items for {need} places.")
    pname = sorted(prot + rest[:need], key=lambda r: (r["house_id"], r["qa_id"]))
    e4 = {r["qa_id"] for r in sorted(pname, key=lambda r: hkey(SEED, "E4", r["qa_id"]))[:E4_SIZE]}
    for r in pname:
        r["in_E4_600"] = r["qa_id"] in e4
        r.pop("_eligible", None)
    pclust = [r for r in pname if r["in_qualifying_cluster"]]

    ids = [r["qa_id"] for r in pname]
    assert len(ids) == PNAME_SIZE and len(set(ids)) == PNAME_SIZE, "P-NAME size or duplicates"
    assert all(r["qa_type"] == "obj_name" for r in pname), "non obj_name item"
    assert {r["qa_id"] for r in prot} <= set(ids), "a protected item is missing"
    all_clust = {r["qa_id"] for r in rows if r["in_qualifying_cluster"]}
    assert all_clust == {r["qa_id"] for r in pclust}, "P-CLUST is not every qualifying-cluster item"
    assert sum(r["in_E4_600"] for r in pname) == E4_SIZE, "E4 flag count"
    return pname, pclust, rows, n_by_name


def stratum_stats(items):
    out = {}
    for lab in (CS, "globally_common", "ambiguous", ""):
        rs = [r for r in items if r["cultural_specificity"] == lab]
        out[lab or "untagged"] = {"items": len(rs), "names": len({r["name_bn"] for r in rs}),
                                  "houses": len({r["house_id"] for r in rs})}
    return out


def write_probe(path, probe, items, dataset_sha, sampling):
    doc = {"probe": probe, "n": len(items), "seed": SEED, "sampling": sampling,
           "dataset_sha256": dataset_sha,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "qa_id_list_sha256": a5.qa_id_list_hash(items),
           "items": [{k: it[k] for k in ITEM_KEYS if k in it and it[k] is not None} for it in items]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    say(f"[write] {path} ({len(items)} items)")


def load_probe_questions(paths):
    qs = set()
    for p in paths:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                qs |= {normalise(i["question_bn"]) for i in json.load(f)["items"]}
    return qs


def cmd_draw(args):
    items = a5.load_items(args.data)
    a5.validate_items(items, args.counts)
    tags = {r["name_id"]: r["cultural_specificity"] for r in a5.read_csv(args.tags)}
    say(f"[tags] {len(tags)} frozen tags from {args.tags}")
    clusters, owner = load_clusters(args.clusters)
    answers = Counter(normalise(it["answer_bn"]) for it in items if it["qa_type"] == "obj_name")
    counts = recount(clusters, answers)
    check_against_asset1(clusters, counts, args.cluster_counts)
    qual = qualifying(clusters, counts)

    pname, pclust, rows, n_by_name = draw_pname(items, tags, clusters, owner, qual)
    out = args.out
    os.makedirs(out, exist_ok=True)
    p1, p2 = os.path.join(out, "P-NAME.json"), os.path.join(out, "P-CLUST.json")
    a5.guard_frozen(p1, pname, args.force)
    a5.guard_frozen(p2, pclust, args.force)
    data_sha = sha256_file(args.data)
    write_probe(p1, "P-NAME", pname, data_sha,
                "protected items in full (D58); the rest by lowest SHA-256('42|P-NAME|<qa_id>'); "
                "in_E4_600 by lowest SHA-256('42|E4|<qa_id>') (D59)")
    write_probe(p2, "P-CLUST", pclust, data_sha, "every P-NAME item in a qualifying cluster")

    per_cluster = {}
    for cid in qual:
        ms = {m["member"]: counts[(cid, m["member"])] for m in clusters[cid]}
        per_cluster[cid] = {"items": sum(ms.values()), "members_at_8_or_more":
                            sum(1 for v in ms.values() if v >= QUAL_MIN_MEMBER), "members": ms}
    below_floor_members = sorted({(r["cluster_member"], n_by_name[r["name_bn"]]) for r in pclust
                                  if r["cultural_specificity"] != CS and n_by_name[r["name_bn"]] < FLOOR_OTHER})
    old_q = load_probe_questions([args.pcore, args.phaz])
    new_q = {normalise(r["question_bn"]) for r in pname} - old_q
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "asset": "ASSET-5B",
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "normalise_py_sha256": sha256_file(os.path.join(here, "normalise.py")),
        "seed": SEED,
        "inputs_sha256": {"dataset": data_sha, "cultural_specificity.csv": sha256_file(args.tags),
                          "clusters_parsed.csv": sha256_file(args.clusters),
                          "cluster_counts.csv": sha256_file(args.cluster_counts)},
        "rules": {"eligible": "obj_name items whose normalise()d gold name has >= 10 obj_name "
                              "answers, or >= 5 if tagged culturally_specific (D42, D57)",
                  "protected": "every eligible culturally_specific item and every item of a "
                               "qualifying-cluster member, taken in full (D58)",
                  "fill": "remaining places from other eligible items by lowest "
                          "SHA-256('42|P-NAME|<qa_id>')",
                  "E4_600": "600 P-NAME items with lowest SHA-256('42|E4|<qa_id>') (D59)",
                  "clusters": "clusters_parsed.csv with D43 applied; qualifies with >= 30 "
                              "obj_name items and >= 2 members at >= 8"},
        "P-NAME": {"n": len(pname), "qa_id_list_sha256": a5.qa_id_list_hash(pname),
                   "protected": dict(Counter(r["protected"] or "fill" for r in pname)),
                   "strata": stratum_stats(pname),
                   "houses": len({r["house_id"] for r in pname}),
                   "E4_600": {"qa_id_list_sha256": a5.qa_id_list_hash([r for r in pname if r["in_E4_600"]]),
                              "strata": stratum_stats([r for r in pname if r["in_E4_600"]]),
                              "cluster_items": sum(1 for r in pname if r["in_E4_600"] and r["in_qualifying_cluster"])}},
        "P-CLUST": {"n": len(pclust), "qa_id_list_sha256": a5.qa_id_list_hash(pclust),
                    "clusters": per_cluster,
                    "members_kept_by_D58_below_the_10_floor": below_floor_members},
        "asset2_wave2": {"P-NAME_questions_not_in_P-CORE_or_P-HAZ": len(new_q),
                         "P-NAME_gold_names": len({r["name_bn"] for r in pname}),
                         "note": "gold names take their English from lexicon.csv canonical_en (D56)"},
        "outputs_sha256": {os.path.basename(p1): sha256_file(p1), os.path.basename(p2): sha256_file(p2)},
    }
    mp = os.path.join(out, "probe_manifest_5B.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    say(f"[write] {mp}")
    s = manifest["P-NAME"]["strata"]
    say()
    say("=" * 70)
    say(f"P-NAME  {len(pname)} items, {manifest['P-NAME']['houses']} houses | protected "
        f"{manifest['P-NAME']['protected']}")
    for lab, v in s.items():
        say(f"        {lab:20s} {v['items']:4d} items, {v['names']:3d} names, {v['houses']:2d} houses")
    say(f"P-CLUST {len(pclust)} items in {len(qual)} clusters; members kept below the 10 floor "
        f"by D58: {len(below_floor_members)}")
    say(f"E4-600  flagged; culturally_specific items in it: "
        f"{manifest['P-NAME']['E4_600']['strata'][CS]['items']}")
    say(f"Wave 2  {len(new_q)} new questions for ASSET-2; gold names come from canonical_en")
    say("=" * 70)


# ----------------------------------------------------------------------------
# Self-test on SYNTHETIC data
# ----------------------------------------------------------------------------
SYN_CLUSTERS = [("রান্নার পাত্র (পট/প্যান)", "হাঁড়ি"), ("রান্নার পাত্র (পট/প্যান)", "পাতিল"),
                ("রান্নার পাত্র (পট/প্যান)", "দেকচি"),
                ("অন্যান্য জোড়া", "টব"), ("অন্যান্য জোড়া", "ফুলদানি"), ("অন্যান্য জোড়া", "ডাস্টবিন"),
                ("বাথরুমের কল / পানি ব্যবস্থা", "বেসিন"), ("বাথরুমের কল / পানি ব্যবস্থা", "বদনা"),
                ("পানি ও পরিবেশনের পাত্র", "বালতি"), ("পানি ও পরিবেশনের পাত্র", "বদনা"),
                ("বৈদ্যুতিক সংযোগ", "সুইচ বোর্ড"), ("বৈদ্যুতিক সংযোগ", "ফিল্টার"),
                ("বৈদ্যুতিক সংযোগ", "কুলা")]


def cmd_selftest(args):
    global G1_READING
    say("#" * 70)
    say("SELF-TEST on SYNTHETIC data. No number printed here is a dataset figure.")
    say("#" * 70)
    tmp = tempfile.mkdtemp(prefix="asset5b_selftest_")
    saved = G1_READING
    try:
        data = os.path.join(tmp, "synthetic_merged.json")
        with open(data, "w", encoding="utf-8") as f:
            json.dump(a4.synthetic_dataset(args.counts), f, ensure_ascii=False)
        counts_csv = os.path.join(tmp, "house_type_counts.csv")
        shutil.copy(args.counts, counts_csv)
        d4 = os.path.join(tmp, "asset4a")
        tags_path = os.path.join(tmp, "cultural_specificity.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            a4.cmd_build(argparse.Namespace(data=data, counts=counts_csv, out=d4, force=False))
            m = a5.read_csv(os.path.join(d4, "asset4a_master.csv"))
            paths = a4.fill_sheets(d4, a4.simulated_votes(m))
            a4.cmd_merge(argparse.Namespace(dir=d4, raters=paths, session=None, freeze=True,
                                            final=tags_path, force=False))
            items = a5.load_items(data)
        ans = Counter(normalise(i["answer_bn"]) for i in items if i["qa_type"] == "obj_name")
        cp = os.path.join(tmp, "clusters_parsed.csv")
        a5.write_csv(cp, [{"cluster_id": c, "member": mm, "surface_forms": mm, "description": ""}
                          for c, mm in SYN_CLUSTERS], ["cluster_id", "member", "surface_forms", "description"])
        cc = os.path.join(tmp, "cluster_counts.csv")
        dropped = {("অন্যান্য জোড়া", "ডাস্টবিন"), ("পানি ও পরিবেশনের পাত্র", "বদনা")}
        a5.write_csv(cc, [{"cluster_id": c, "member": mm,
                           "obj_name_count": 0 if (c, mm) in dropped else ans.get(normalise(mm), 0)}
                          for c, mm in SYN_CLUSTERS], ["cluster_id", "member", "obj_name_count"])
        G1_READING = sorted([(ans[normalise("হাঁড়ি")] + ans[normalise("পাতিল")] + ans[normalise("দেকচি")], 3),
                             (ans[normalise("বেসিন")] + ans[normalise("বদনা")], 2),
                             (ans[normalise("সুইচ বোর্ড")] + ans[normalise("ফিল্টার")] + ans[normalise("কুলা")], 2)],
                            reverse=True)
        out = os.path.join(tmp, "out")
        dargs = argparse.Namespace(data=data, counts=counts_csv, tags=tags_path, clusters=cp,
                                   cluster_counts=cc, out=out, pcore=None, phaz=None, force=False)
        say("[selftest] --- draw ---")
        cmd_draw(dargs)
        doc = json.load(open(os.path.join(out, "P-NAME.json"), encoding="utf-8"))
        pn = doc["items"]
        pc = json.load(open(os.path.join(out, "P-CLUST.json"), encoding="utf-8"))["items"]
        names = Counter(r["name_bn"] for r in pn)
        N = normalise
        assert len(pn) == PNAME_SIZE and len({r["qa_id"] for r in pn}) == PNAME_SIZE
        assert names[N("কুলা")] == ans[N("কুলা")], "a below-floor cluster member was not kept"
        assert names[N("বটি")] == ans[N("বটি")] and names[N("গামছা")] == ans[N("গামছা")], "CS census"
        assert names[N("দরজা")] == 0, "a below-floor name outside clusters entered P-NAME"
        assert {r["cluster_id"] for r in pc} == {"রান্নার পাত্র (পট/প্যান)", "বাথরুমের কল / পানি ব্যবস্থা",
                                                 "বৈদ্যুতিক সংযোগ"}
        assert all(r["cluster_id"] == "বাথরুমের কল / পানি ব্যবস্থা" for r in pc if r["name_bn"] == N("বদনা"))
        assert sum(r["in_E4_600"] for r in pn) == E4_SIZE
        say("[selftest] OK  1,000 items; census of culturally specific and cluster items; floors hold")
        say("[selftest] OK  D43: বদনা only in the bathroom cluster; E4 flags 600 items")

        say("[selftest] --- draw again: identical ---")
        with contextlib.redirect_stdout(io.StringIO()):
            cmd_draw(dargs)
        again = json.load(open(os.path.join(out, "P-NAME.json"), encoding="utf-8"))
        assert again["qa_id_list_sha256"] == doc["qa_id_list_sha256"]
        say("[selftest] OK  a repeat draw reproduces P-NAME exactly")

        def must_stop(label, fn):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fn()
                raise AssertionError(f"did not stop: {label}")
            except SystemExit as e:
                say(f"[selftest] OK  {label:34s} -> {str(e)[:58]}...")

        bad_cc = os.path.join(tmp, "bad_counts.csv")
        rows = a5.read_csv(cc)
        rows[0]["obj_name_count"] = str(int(rows[0]["obj_name_count"]) + 1)
        a5.write_csv(bad_cc, rows, ["cluster_id", "member", "obj_name_count"])
        must_stop("recount differs from ASSET-1", lambda: cmd_draw(argparse.Namespace(**dict(vars(dargs), cluster_counts=bad_cc))))
        good = G1_READING
        G1_READING = sorted(good + [(99, 2)], reverse=True)
        must_stop("clusters differ from the G1 reading", lambda: cmd_draw(dargs))
        G1_READING = good
        two = os.path.join(tmp, "two.csv")
        a5.write_csv(two, [{"cluster_id": c, "member": mm, "surface_forms": mm + ("|বেসিন" if mm == "ফিল্টার" else ""),
                            "description": ""} for c, mm in SYN_CLUSTERS],
                     ["cluster_id", "member", "surface_forms", "description"])
        must_stop("a surface form in two clusters", lambda: cmd_draw(argparse.Namespace(**dict(vars(dargs), clusters=two))))
        doc["items"] = doc["items"][1:]
        with open(os.path.join(out, "P-NAME.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        must_stop("frozen P-NAME would change", lambda: cmd_draw(dargs))
    finally:
        G1_READING = saved
        shutil.rmtree(tmp, ignore_errors=True)
    say()
    say("SELF-TEST PASSED: rules hold, the draw repeats exactly, and every cross-check")
    say("and the frozen-probe guard stop the run when they should.")


def main():
    ap = argparse.ArgumentParser(description="ASSET-5B: draw P-NAME and P-CLUST")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s = sub.add_parser("draw")
    s.add_argument("--data", required=True)
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s.add_argument("--tags", default="results/assets/cultural_specificity.csv")
    s.add_argument("--clusters", default="results/assets/clusters_parsed.csv")
    s.add_argument("--cluster-counts", dest="cluster_counts", default="results/assets/cluster_counts.csv")
    s.add_argument("--pcore", default="results/assets/P-CORE.json")
    s.add_argument("--phaz", default="results/assets/P-HAZ.json")
    s.add_argument("--out", default="results/assets")
    s.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "draw": cmd_draw}[args.cmd](args)


if __name__ == "__main__":
    main()
