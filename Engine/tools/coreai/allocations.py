"""What the GPU allocated while a command ran, by who allocated it: the reading of a trace recorded with Instruments'
Core AI template and Metal Resource Events — every Metal resource with its size, its life and its backtrace, and the
device's allocated size over time. No Xcode project, no attach:

    xcrun xctrace record --template 'Core AI' --instrument 'Metal Resource Events' --output run.trace \\
        --launch -- App/.build/release/siliconed check qwen-image-2.1 memory
    python3 Engine/tools/coreai/allocations.py run.trace [MIN_MIB]

It prints the device's peak, then each allocator (the first frame that names one) with the sizes it asked for, then
the allocations of MIN_MIB or more (64) in time. How the padded copies of short-rowed weights were found
(`MPSGraphDelegateKernel.convertCoreAIRawView`, 90 buffers held a whole inference). VM Tracker and Allocations need to
attach to the process, which a session without the right cannot.
"""
import collections
import subprocess
import sys
import xml.etree.ElementTree as ET

M = 1 << 20
ALLOCATORS = ("convertCoreAIRawView", "MemrefBufferizer", "MPSAutoCache", "WeightWindow", "SessionPlan", "MPSNDArray")


def rows(trace, schema):
    xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                          f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'], capture_output=True, check=True).stdout
    node = ET.fromstring(xml).find("node")
    columns = [c.find("mnemonic").text for c in node.find("schema").findall("col")]
    seen = {}

    def value(e):
        if e.tag == "sentinel": return None
        if "ref" in e.attrib: return seen[e.attrib["ref"]]
        if e.tag == "tagged-backtrace":
            v = [f.attrib.get("name", "?") for f in e.iter("frame")]
        else:
            for f in e.iter():
                if f is not e and "id" in f.attrib: seen[f.attrib["id"]] = (f.text, f.attrib.get("fmt"))
            v = (e.text, e.attrib.get("fmt"))
        if "id" in e.attrib: seen[e.attrib["id"]] = v
        return v

    for r in node.findall("row"):
        yield dict(zip(columns, [value(e) for e in r]))


trace, least = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
held = [(int(r["start"][0]) / 1e9, int(r["current-allocated-size"][0])) for r in rows(trace, "metal-current-allocated-size")]
if held:
    at, peak = max(held, key=lambda x: x[1])
    print(f"the device held at most {peak / M:.1f} MiB (at {at:.3f} s), {held[-1][1] / M:.1f} MiB at the end")
by = collections.defaultdict(lambda: [0, 0, collections.Counter()])
large = []
for r in rows(trace, "metal-resource-allocations"):
    size = int((r["resource-size"] or ("0",))[0] or 0)
    frames = r["backtrace"] if isinstance(r["backtrace"], list) else []
    who = next((f for f in frames if any(k in f for k in ALLOCATORS)), " < ".join(frames[2:5]))[:96]
    event = (r["event-type"] or ("", ""))[1]
    if event == "Allocation":
        entry = by[who]
        entry[0] += 1; entry[1] += size; entry[2][size] += 1
    if size >= least * M: large.append((int(r["timestamp"][0]) / 1e9, size, event, (r["duration"] or (None, "—"))[1], who))
print("\nby allocator")
for who, (count, total, sizes) in sorted(by.items(), key=lambda x: -x[1][1]):
    if total < M: continue
    print(f"{total / M:10.1f} MiB  {count:5d}  {who}")
    for size, n in sorted(sizes.items(), reverse=True)[:8]:
        if size >= M: print(f"{'':20}{n:4d} × {size / M:9.1f} MiB")
print(f"\n{least} MiB or more, in time")
for at, size, event, life, who in large:
    print(f"{at:9.3f} s  {event:12} {size / M:9.1f} MiB  life {life:10}  {who}")
