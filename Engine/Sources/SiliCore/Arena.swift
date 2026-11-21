import CoreVideo
import Foundation
import IOSurface
import Metal

/// The arena: one allocation at the start, at a size computed before — a shared placement `MTLHeap` — and every tensor
/// an `MTLBuffer` of its own placed in it at an offset. The function's inputs and outputs are those buffers, whole, so
/// no copy is born at either boundary.
///
/// **Every tensor starts on a page** (16 KiB): Core AI's GPU copies, at every call, an input whose buffer does not
/// (measured 17/11 on the M1 Pro, a transformer block's three functions on three layers' weights). The heap's own
/// alignment, 256 bytes, had put two of layer 1's tensors on 8 KiB where layers 0 and 2 had them on 16 or more, and
/// every pass on layer 1 held 41.4 MiB more than the others: those two tensors are 41.25. On pages, the block's three
/// functions together went from 979 to 493 MiB at the peak and a pass from 676 to 574 ms; the feed-forward function
/// alone from 519 to 272 MiB — and the ~120 MiB each of its calls from the second on took and gave back went with it:
/// they were these copies, not Core AI's own. (`RunTests`: `anInputOffThePageIsCopiedAtEveryCall`.)
///
/// **Why not one `MTLBuffer` with views at offsets** (measured 17/11 with views on 256 bytes, before the pages; not
/// measured since, and the copies above may be the whole of it): a call cost with the buffer's extent past its views
/// (on 30 layers' weights in order, ~520 ms for layer 0 to ~81 for layer 29, +15 ms by 112 MiB, ~7.8 GB/s), the device
/// held what the footprint did not show (+854 MiB of `currentAllocatedSize` for +270 of footprint), and with the
/// activations first 136 of Core AI's command buffers failed `kIOGPUCommandBufferCallbackErrorOutOfMemory`, said on
/// stderr only: `encode` throws nothing, the results are silently not computed.
///
/// The heap's hazard tracking changes nothing (tracked or not, within 0.2 MiB). An idle GPU slows the next call: 94.5 ms
/// after a pause of 20 ms, 120 after 200, for 72.7 back to back (the feed-forward function, before the pages).
/// The ring's reader (`Ring`) writes into its buffers' contents from a thread of its own; what it takes in flight (a
/// group kept, `grow`) is found under `lock`, as below.
///
/// **What the ANE reads or writes is a surface of its own** (`take(surface:)`): the ANE called by `encode` on the heap's
/// buffers held one call in 17 to 33 for 560 to 660 ms (a feed-forward half at 1024 rows on the M1 Pro: median 43 to 50
/// ms, mean 58 to 81), by `run` on the same buffers none but 49 to 51 ms, by `run` on `IOSurface`s none and 31 to 32,
/// its result the same to the bit. Core AI takes a surface only at the tensor's exact shape (its width the last
/// dimension, its height the product of the others; any other, a fatal error): one surface a tensor, never views into
/// one. The GPU reads the same bytes through an `MTLBuffer` over them (`bytesNoCopy`) as it reads the heap's (68.7 ms
/// against 69.0, the same to the bit; `RunTests`: `theGPUReadsASurfaceAsItReadsTheHeap`), and the process's footprint
/// counts them as they are touched. A surface of n bytes allocates `aligned(n)` (read 17/11 from 2 bytes to 42 MiB),
/// so the arena's count stays exact.
///
/// **Heaps added while the stage runs** (`grow`): a group the stage keeps once it has read what Core AI holds (`Stage`)
/// is placed in a heap of its own, taken by the reader's thread while the loop's reads the arena's buffers — every
/// buffer, pointer and surface found under `lock`.
public final class Arena: @unchecked Sendable {
    private let device: MTLDevice
    /// The heaps, the first the arena's own (`init`), the others added (`grow`): each its first key and the bytes taken.
    private var heaps: [(heap: MTLHeap, base: Int, used: Int)]
    /// The heaps being allocated (`grow`), counted already.
    private var growing = 0
    private var placed: [Int: MTLBuffer] = [:]
    /// The parts made (`part`), by their first byte's key and their length; made during calls, under `lock`.
    private var parts: [Int: [Int: MTLBuffer]] = [:]
    private let lock = NSLock()
    /// Each buffer's bytes, read once as it is taken (`pointer`): `contents()` retains and autoreleases its buffer
    /// (measured 19/11 on the M5 Pro), so a write by the CPU from a thread whose pool is not drained — the command
    /// line's main thread, waiting on the stages it opens — kept the buffer and with it the whole heap past the arena
    /// (a 6 MiB tensor written on the main thread held a stage's heap of 24.7 GiB after `close`). Read in a pool here, a
    /// pointer never touches the buffer again (`RunTests`: `aStageLetGoLeavesNothing`).
    private var bytes: [Int: UnsafeMutableRawPointer] = [:]
    /// The bytes taken in the first heap.
    public var used: Int { lock.withLock { heaps[0].used } }
    /// The surfaces, by their keys: past the heap's offsets.
    private var surfaces: [Int: IOSurface] = [:]
    /// The ranges wired (`wire`), unwired when the arena goes; what every arena of the process holds wired.
    private var wired: [(UnsafeMutableRawPointer, Int)] = []
    nonisolated(unsafe) private static var wiredAll = 0
    public static var wiredBytes: Int { lock.lock(); defer { lock.unlock() }; return wiredAll }
    /// What the arena holds apart from its heaps, the surfaces and the tensors `take(apart:)`: `aligned` of each, what
    /// they allocate.
    public var apartBytes: Int { lock.withLock { apart } }
    private var apart = 0
    /// The next key: past the first heap's offsets, each tensor apart, each surface and each heap added takes its bytes'
    /// worth of keys.
    private var top: Int

    /// What a tensor of `n` bytes takes in the arena: its pages, since it starts on one (below). Metal places a buffer of
    /// n bytes in n (`heapBufferSizeAndAlign`: the length itself, aligned on 256; read 17/11 on the M1 Pro), so an arena
    /// counted by these is the sum of what it holds, exactly. Every row of the functions' tensors is a multiple of 64
    /// bytes, which is what Core AI's GPU binds as it lies (the former engine: any other row it copies at every run,
    /// padded to 64).
    public static func aligned(_ n: Int) -> Int { (n + page - 1) / page * page }

    /// Where a tensor may start: a page, or Core AI copies it at every call (above).
    public static let page = Int(getpagesize())

    private static func descriptor(_ bytes: Int) -> MTLHeapDescriptor {
        let d = MTLHeapDescriptor()
        d.type = .placement
        d.storageMode = .shared
        // Tracked: Metal orders the stream's calls by the buffers they read and write, so a call is put on the stream while
        // the one before runs (`Block`); untracked, a call read its input before the call before wrote it (18/11).
        d.hazardTrackingMode = .tracked
        d.size = max(bytes, page)
        return d
    }

    /// What Metal allocates for an arena of `bytes` (counted by `aligned`): a placement heap is the size asked rounded to
    /// a page (read 17/11 from 1 byte to 4 GB), and the footprint holds all of it from its allocation on (17/11: 6125.7
    /// MiB held for 6076.8 taken, the slack of an older count included). P's term, known before anything is allocated.
    public static func allocation(bytes: Int) -> Int { aligned(max(bytes, page)) }

    /// `bytes`: the sum of `aligned` of every tensor that will be taken; `apart`, of those taken apart from the heap
    /// (`take(apart:)`; the surfaces are their own). Since each is a whole number of pages, the heap and they allocate
    /// `allocation(bytes)` together, as the heap alone did.
    public init(_ device: MTLDevice, bytes: Int, apart: Int = 0) throws {
        self.device = device
        let d = Self.descriptor(bytes - apart)
        guard let h = device.makeHeap(descriptor: d) else { throw Failure("no heap of \(d.size) bytes") }
        heaps = [(h, 0, 0)]
        top = h.size
        Self.lock.lock(); Self.held += h.size; Self.lock.unlock()
    }

    /// **A heap added, of `bytes`** (counted by `aligned`, as the arena's): its index, for `take(_:in:)`. What it takes is
    /// `allocation(bytes:)`, held from now on.
    /// Counted in `allocated` from before Metal allocates it: the footprint holds a heap from inside `makeHeap`, and a
    /// reading between the two would take the heap for Core AI's share (`Stage.read`).
    public func grow(bytes: Int) throws -> Int {
        let d = Self.descriptor(bytes)
        lock.withLock { growing += d.size }
        guard let h = device.makeHeap(descriptor: d) else {
            lock.withLock { growing -= d.size }
            throw Failure("no heap of \(d.size) bytes")
        }
        Self.lock.lock(); Self.held += h.size; Self.lock.unlock()
        return lock.withLock {
            growing -= d.size
            heaps.append((h, top, 0))
            top += h.size
            return heaps.count - 1
        }
    }

    /// What Metal allocated for the arena: its heaps and what lies apart.
    public var allocated: Int { lock.withLock { heaps.reduce(apart + growing) { $0 + $1.heap.size } } }

    deinit {
        for (p, n) in wired { munlock(p, n) }
        let unwired = wired.reduce(0) { $0 + $1.1 }
        // Its heaps weakly held after it: no heap of a stage let go may stay held. Measured 18/11: emptied as it is let go
        // (`setPurgeableState(.empty)`), a heap Core AI still held left 1 to 2 s later; its exit apart, at once.
        Self.lock.lock()
        Self.wiredAll -= unwired
        Self.gone.removeAll { $0.heap == nil }
        for h in heaps { Self.held -= h.heap.size; Self.gone.append(Gone(heap: h.heap, size: h.heap.size)) }
        Self.lock.unlock()
    }

    private final class Gone { weak var heap: MTLHeap?; let size: Int; init(heap: MTLHeap, size: Int) { self.heap = heap; self.size = size } }
    nonisolated(unsafe) private static var gone: [Gone] = []
    /// The heaps of arenas let go that something else still holds.
    public static var lingering: Int { lock.lock(); defer { lock.unlock() }; return gone.filter { $0.heap != nil }.reduce(0) { $0 + $1.size } }

    /// The bytes of the arenas alive in the process.
    nonisolated(unsafe) private static var held = 0
    private static let lock = NSLock()
    public static var alive: Int { lock.lock(); defer { lock.unlock() }; return held }

    /// A tensor of `bytes`: its buffer placed at the returned offset, the tensor's name in the arena. The buffer is its
    /// pages whole, which the arena holds anyway: a function asks at least `minimumByteCount` of an argument, its rows
    /// rounded up (a `[130]` f32 asks 576 bytes for 520), and copies at every call a buffer that is shorter.
    public func take(_ bytes: Int) -> Int { take(bytes, in: 0) }

    /// A tensor of `bytes` in heap `index` (`grow`): its key.
    public func take(_ bytes: Int, in index: Int) -> Int {
        let length = Self.aligned(max(bytes, 1))
        let sa = device.heapBufferSizeAndAlign(length: length, options: .storageModeShared)
        let align = max(sa.align, Self.page)
        lock.lock(); defer { lock.unlock() }
        let (heap, base, used) = heaps[index]
        let offset = (used + align - 1) / align * align
        guard let b = heap.makeBuffer(length: length, options: .storageModeShared, offset: offset) else {
            preconditionFailure("arena overflow: \(offset + sa.size) > \(heap.size)")
        }
        placed[base + offset] = b
        self.bytes[base + offset] = autoreleasepool { b.contents() }
        heaps[index].used = offset + sa.size
        return base + offset
    }

    /// **A tensor in a buffer of its own, apart from the heap**: a stage's exit, what its last call writes. A function
    /// holds the buffers of its last call, inputs and outputs, until it is called again or let go, whatever else is
    /// called (measured 18/11 on the M1 Pro, three stages of a model in a row, each let go: one or two buffers of the last
    /// call still alive 5 s later, until the next stage's calls; a 1 MiB output kept a 6 GiB heap, and the next stage
    /// came over it; on the M5 Pro, the heap gone at once with its function; `RunTests`:
    /// `aFunctionHoldsItsLastCallsBuffers`). Apart, what it holds is the tensor's pages, not the heap.
    public func take(apart bytes: Int) -> Int {
        guard let b = device.makeBuffer(length: Self.aligned(bytes), options: .storageModeShared) else { preconditionFailure("no buffer of \(bytes) B") }
        lock.lock(); defer { lock.unlock() }
        let key = top
        placed[key] = b
        self.bytes[key] = autoreleasepool { b.contents() }
        (apart, top) = (apart + Self.aligned(bytes), top + Self.aligned(bytes))
        return key
    }

    /// A tensor of `shape`, `element` bytes each, as a surface of its own and the GPU's buffer over it (above): its key.
    public func take(surface shape: [Int], element: Int) -> Int {
        let shape = Layout.trimmed(shape)
        let width = shape.last ?? 1, height = shape.dropLast().reduce(1, *), n = width * height * element
        let format = switch element {
        case 1: kCVPixelFormatType_OneComponent8
        case 2: kCVPixelFormatType_OneComponent16Half
        default: kCVPixelFormatType_OneComponent32Float
        }
        guard let s = IOSurface(properties: [.width: width, .height: height, .bytesPerElement: element, .bytesPerRow: width * element, .pixelFormat: format]) else {
            preconditionFailure("no surface of \(shape) × \(element) B")
        }
        precondition(IOSurfaceGetBytesPerRow(s) == width * element && IOSurfaceGetAllocSize(s) == Self.aligned(n),
                     "a surface of \(shape): \(IOSurfaceGetBytesPerRow(s)) B a row, \(IOSurfaceGetAllocSize(s)) B for \(Self.aligned(n))")
        s.lock(options: [], seed: nil)
        let base = s.baseAddress
        s.unlock(options: [], seed: nil)
        guard let b = device.makeBuffer(bytesNoCopy: base, length: Self.aligned(n), options: .storageModeShared, deallocator: nil) else {
            preconditionFailure("no buffer over a surface of \(shape)")
        }
        lock.lock(); defer { lock.unlock() }
        let key = top
        surfaces[key] = s; placed[key] = b; bytes[key] = base
        (apart, top) = (apart + Self.aligned(n), top + Self.aligned(n))
        return key
    }

    /// **A per-row tensor of `bytes` as one surface the ANE takes a unit at a time**: declared `width` × `height` (a
    /// unit's rows: `[unit, n]`, or `[1, unit]` one value a row), its allocation every row (`.allocSize`); the ANE given
    /// unit i by its byte offset, the GPU any rows by a buffer over them (`part`). Measured 19/11 on the M5 Pro: the ANE
    /// refuses an offset into a surface taller than the view (« IOSurface height … does not match ») and takes a unit of
    /// a surface one unit high allocated four, the same bits, 0.0 MiB a call; the GPU reads its rows as fast as the heap's.
    public func take(units bytes: Int, width: Int, height: Int, element: Int) -> Int {
        let n = Self.aligned(bytes)
        let format = element == 2 ? kCVPixelFormatType_OneComponent16Half : element == 1 ? kCVPixelFormatType_OneComponent8 : kCVPixelFormatType_OneComponent32Float
        guard let s = IOSurface(properties: [.width: width, .height: height, .bytesPerElement: element, .bytesPerRow: width * element,
                                             .allocSize: n, .pixelFormat: format]) else {
            preconditionFailure("no surface of \(height) × \(width) × \(element) B over \(bytes) B")
        }
        precondition(IOSurfaceGetBytesPerRow(s) == width * element && IOSurfaceGetAllocSize(s) >= n,
                     "a surface of \(height) × \(width): \(IOSurfaceGetBytesPerRow(s)) B a row, \(IOSurfaceGetAllocSize(s)) B for \(n)")
        s.lock(options: [], seed: nil)
        let base = s.baseAddress
        s.unlock(options: [], seed: nil)
        guard let b = device.makeBuffer(bytesNoCopy: base, length: n, options: .storageModeShared, deallocator: nil) else {
            preconditionFailure("no buffer over a surface of \(bytes) B")
        }
        // What the surface holds, counted: macOS may give more than asked, and P is what is held.
        let held = IOSurfaceGetAllocSize(s)
        lock.lock(); defer { lock.unlock() }
        let key = top
        surfaces[key] = s; placed[key] = b; self.bytes[key] = base
        (apart, top) = (apart + held, top + held)
        return key
    }

    /// **Tensor `key`'s pages wired** (`mlock`), `bytes` of them: what macOS may not compress. A weight the loop reads once
    /// a cycle is, for the system, a page left alone between two reads: under pressure it compresses it, and every call
    /// that reads it pays the decompression (measured 20/11 on the M1 Pro, Z Compacte 1024² at B = 10, a browser open:
    /// 2.1 of 6.3 GiB of kept weights compressed, the free memory at 60 MiB, the ANE's calls up to +39 % and the GPU's
    /// beside them +26 %, the GPU's own steps, which read no weight, unchanged; wired, every cycle as fast as the ring's,
    /// 83 – 86 → 79.9 s, no page swapped out). Unwired with the arena. False where the system refuses (`Memory.wireLimit`).
    @discardableResult
    public func wire(_ key: Int, bytes: Int) -> Bool {
        let p = pointer(key), n = Self.aligned(bytes)
        guard mlock(p, n) == 0 else { return false }
        lock.withLock { wired.append((p, n)) }
        Self.lock.lock(); Self.wiredAll += n; Self.lock.unlock()
        return true
    }

    /// The surface of a tensor `take(surface:)`, nil for any other.
    public func surface(_ key: Int) -> IOSurface? { lock.withLock { surfaces[key] } }

    /// What the first heap holds: what Metal allocated (`allocation`).
    public var size: Int { lock.withLock { heaps[0].heap.size } }

    public func buffer(_ offset: Int) -> MTLBuffer { lock.withLock { placed[offset]! } }

    /// **Part of a tensor of the heap as a buffer of its own**, over the same bytes: `bytes` from byte `from` (a page)
    /// of the tensor `key`, its pages whole as far as the tensor goes. A function reads part of its buffer more slowly
    /// than a buffer whole (measured 19/11 on the M5 Pro, `layoutProbe`: a 32 MiB f16 view at the start of a buffer 4 ×
    /// its size 1.45 – 1.53 ms a call for 1.08, from a page 1 MiB in 4.4; a 1024-row tile of a 4128 × 3840 bf16 stream, its product read and written as views
    /// of the stream's buffers 3.23 ms, as buffers of their own over the same bytes 2.99, the same bits). Made once,
    /// kept with the arena; a tensor apart's or a surface's over its bytes (`bytesNoCopy`), which holds no heap.
    public func part(_ key: Int, from: Int, bytes: Int) -> MTLBuffer? {
        guard from % Self.page == 0 else { return nil }
        lock.lock(); defer { lock.unlock() }
        let base = placed[key]!
        let length = min(Self.aligned(bytes), base.length - from)
        if let b = parts[key + from]?[length] { return b }
        let made = if let heap = base.heap { heap.makeBuffer(length: length, options: .storageModeShared, offset: base.heapOffset + from) }
            else { device.makeBuffer(bytesNoCopy: self.bytes[key]! + from, length: length, options: .storageModeShared, deallocator: nil) }
        guard let b = made else { preconditionFailure("no buffer of \(length) B over the tensor \(key) from \(from)") }
        parts[key + from, default: [:]][length] = b
        return b
    }

    public func pointer(_ offset: Int) -> UnsafeMutableRawPointer { lock.withLock { bytes[offset]! } }
}

/// What the engine throws: a sentence.
public struct Failure: Error, CustomStringConvertible {
    public let description: String
    public init(_ d: String) { description = d }
}

/// **A stage refused under B before anything is allocated** (`Stage.plan`, `Stage.choose`): what it needs at its
/// soberest, the margin counted, against B — what a client says as a lack of memory, not as a failure.
public struct OverBudget: Error, CustomStringConvertible {
    public let needed, budget: Int
    public let description: String
}
