import Foundation

// **What travels between the app and its engine's process** (`silicore-worker`), over two pipes: the worker's standard
// input carries the app's calls, its standard output the replies. The frames are the ones two Macs exchange
// (`Frames.swift`): a message is JSON, a blob goes in 64 KiB pieces. **On a pipe a blob goes before the message that
// names it**, written in that order by one writer (`WorkerPipe`): when the message arrives, its blobs are whole.
//
//     both ways      hello(protocol), first: an app and a worker of two builds refuse each other there
//     app → worker   render(id, order) · write(id, order) · erase(id) · cancel(id)
//                    the images: blobs, before their call
//     worker → app   event(id, …) · written(id, …) · erased(id, bytes) · finished(id) · failed(id, error)
//                    an image or a preview: a blob, before its event
//
// **One piece of work at a time**: the worker runs its calls in their order, a call waiting for the one before it. A
// cancelled call still ends with `finished` or `failed`: until then the worker computes, and the app's fuse watches.

/// **The first message each way: the protocol its writer speaks** (`version`). The app refuses a worker of another
/// version (`EngineError.engineMismatch`), the worker an app of another by exiting (status 66), before any call.
/// `version` goes up with any change to what crosses below; the hello's own shape never changes — every build reads it
/// first.
package struct WorkerHello: Codable, Sendable, Equatable {
    package static let version = 6
    package var protocolVersion: Int

    package init(protocolVersion: Int = WorkerHello.version) { self.protocolVersion = protocolVersion }

    /// The version a first message says, `nil` when it is not a hello (a build from before the hello).
    package static func version(of message: Data) -> Int? {
        (try? JSONDecoder().decode(WorkerHello.self, from: message))?.protocolVersion
    }
}

/// **A call of the app's.**
package enum WorkerCall: Codable, Sendable, Equatable {
    case render(id: UInt32, RenderOrder)
    case write(id: UInt32, WriteOrder)
    /// The compilations erased (`CoreAI.erase`), in the order of the calls: after the work before it, before the work after.
    case erase(id: UInt32)
    case cancel(id: UInt32)
}

/// **A reply of the worker's.**
package enum WorkerReply: Codable, Sendable {
    /// One of a render's events, as the engine emitted it.
    case event(id: UInt32, WorkerEvent)
    /// One of the writer's events.
    case written(id: UInt32, PromptWriter.Event)
    /// What an erasure freed, in bytes, before its `finished`.
    case erased(id: UInt32, bytes: Int)
    /// The call is over, its last event sent.
    case finished(id: UInt32)
    case failed(id: UInt32, WorkerFailure)
}

/// What a call fails with: the engine's one error, or the writer's own refusals.
package enum WorkerFailure: Codable, Sendable, Equatable {
    case engine(EngineError)
    case writer(PromptWriter.Failure)

    package init(_ error: any Error) {
        if let w = error as? PromptWriter.Failure { self = .writer(w) } else { self = .engine(EngineError(error)) }
    }

    /// The error a consumer of the stream receives.
    package var error: any Error {
        switch self {
        case .engine(let e): e
        case .writer(let w): w
        }
    }
}

/// An image carried by a blob: `ImageRGB`'s floats, little-endian, then its RGBA bytes when it has them.
package struct WorkerImage: Codable, Sendable, Equatable {
    package var blob: UInt32
    package var width: Int, height: Int
    package var alpha: Bool

    /// The blob's bytes: the floats as they are in memory (an image goes back bit for bit, and its PNG with it).
    package static func bytes(_ image: ImageRGB) -> Data {
        var d = image.pixels.withUnsafeBufferPointer { Data(buffer: $0) }
        if let rgba = image.rgba { d.append(contentsOf: rgba) }
        return d
    }

    package func image(_ data: Data) throws -> ImageRGB {
        let values = 3 * width * height
        guard width > 0, height > 0, data.count == values * 4 + (alpha ? 4 * width * height : 0) else {
            throw EngineError.internalFailure(component: "worker", detail: "an image of \(data.count) bytes for \(width)×\(height)")
        }
        let pixels = [Float](unsafeUninitializedCapacity: values) { buffer, count in
            _ = data.prefix(values * 4).copyBytes(to: buffer)
            count = values
        }
        let rgba = alpha ? [UInt8](data.suffix(4 * width * height)) : nil
        return ImageRGB(pixels: pixels, height: height, width: width, rgba: rgba)
    }
}

/// **A render, asked of the worker**: a `Request` and its seeds, the model by its identifier in a library, the
/// Neural Engine switch the app plans under.
package struct RenderOrder: Codable, Sendable, Equatable {
    package var prompt: String
    package var width: Int, height: Int
    package var seed: UInt64
    package var steps: Int?
    package var loras: [LoRAEntry]
    package var strength: Double
    package var previews: Bool
    package var detail: Detail
    package var solver: Solver
    package var variations: [Variation]
    /// The starting image, already fitted to the format (`Request.image`); the references, as they are.
    package var image: WorkerImage?
    package var references: [WorkerImage]
    package var seeds: [UInt64]
    package var model: String
    package var library: WorkerLibrary
    /// The app's Neural Engine switch (`CoreAI.neuralEngine`), the process's in the worker: every render plans under it.
    package var neuralEngine: Bool
    /// **Where the render keeps its latents between its DiT and its decoder** (`RenderLoop.keeping`): the app's file for
    /// this call, which a worker handed the call again after the fuse stopped the one before finds and decodes alone.
    package var keeping: String?

    /// The order of `request`, its images numbered from `firstBlob`; the blobs to send before it, in order.
    package static func of(_ request: Request, seeds: [UInt64], model: String, library: Library, neuralEngine: Bool,
                           keeping: URL? = nil, firstBlob: UInt32) -> (order: RenderOrder, blobs: [(UInt32, Data)]) {
        var blobs: [(UInt32, Data)] = []
        func carried(_ image: ImageRGB) -> WorkerImage {
            let id = firstBlob &+ UInt32(blobs.count)
            blobs.append((id, WorkerImage.bytes(image)))
            return WorkerImage(blob: id, width: image.width, height: image.height, alpha: image.rgba != nil)
        }
        let order = RenderOrder(prompt: request.prompt, width: request.width, height: request.height, seed: request.seed,
                                steps: request.steps, loras: request.loras, strength: request.strength,
                                previews: request.previews, detail: request.detail, solver: request.solver,
                                variations: request.variations,
                                image: request.image.map(carried), references: request.references.map(carried),
                                seeds: seeds, model: model, library: WorkerLibrary(library), neuralEngine: neuralEngine,
                                keeping: keeping?.path)
        return (order, blobs)
    }

    /// The request again, its images read from `blobs`.
    package func request(_ blobs: [UInt32: Data]) throws -> Request {
        func read(_ w: WorkerImage) throws -> ImageRGB {
            guard let data = blobs[w.blob] else {
                throw EngineError.internalFailure(component: "worker", detail: "the image of blob \(w.blob) never came")
            }
            return try w.image(data)
        }
        // The format first: the starting image is already at it, and comes back bit for bit (`Request.fit`).
        var r = Request(prompt, width: width, height: height, seed: seed, steps: steps, loras: loras,
                        image: try image.map(read), strength: strength, previews: previews)
        r.references = try references.map(read)
        r.detail = detail
        r.solver = solver
        r.variations = variations
        return r
    }

    /// The blobs it names.
    package var blobs: [UInt32] { (image.map { [$0.blob] } ?? []) + references.map(\.blob) }
}

/// **A gesture of the Prompt Lab's writer**, asked of the worker. Describe's image goes as the file's bytes, read
/// there as the writer reads it (`PromptWriterImage`).
package struct WriteOrder: Codable, Sendable, Equatable {
    package enum Input: Codable, Sendable, Equatable {
        case rewrite(text: String)
        case describe(imageBlob: UInt32)
    }
    package var input: Input
    package var instruction: String
    package var library: WorkerLibrary
    package var neuralEngine: Bool

    package var blobs: [UInt32] {
        if case .describe(let b) = input { return [b] }
        return []
    }
}

/// A library by its two folders (`Library.root`, `Library.local`).
package struct WorkerLibrary: Codable, Sendable, Equatable {
    package var root: String, local: String
    package init(_ library: Library) { root = library.root.path; local = library.local.path }
    package var library: Library { Library(root: URL(fileURLWithPath: root), local: URL(fileURLWithPath: local)) }
}

// MARK: - The events

/// **`Engine.Event` on the wire**: the same cases, an image and a preview's pixels carried by a blob.
package enum WorkerEvent: Codable, Sendable {
    case start(startStep: Int, startSigma: Float, evaluations: Int, seeds: [UInt64], stages: [Engine.Stage])
    case stage(Engine.Stage, image: Int)
    case text(tokens: Int, tokenizer: Double, encoder: Double, cached: Bool)
    case encoding(seconds: Double)
    case step(image: Int, index: Int, total: Int, sigma: Float, seconds: Double, latentGridHeight: Int, latentGridWidth: Int)
    case preview(image: Int, index: Int, total: Int, height: Int, width: Int, blob: UInt32)
    case decoding(seconds: Double)
    case image(index: Int, WorkerRender)
    case unusable(image: Int, stage: Engine.Stage)
    case warning(String)
    case held([Engine.Held])
    case memory(Engine.MemorySeries)

    /// The event on the wire, and the blob that goes before it.
    package static func of(_ event: Engine.Event, blob id: UInt32) -> (event: WorkerEvent, blob: Data?) {
        switch event {
        case .start(let p):
            return (.start(startStep: p.denoising.startStep, startSigma: p.denoising.startSigma,
                           evaluations: p.denoising.evaluations, seeds: p.seeds, stages: p.stages), nil)
        case let .stage(s, image): return (.stage(s, image: image), nil)
        case let .text(tokens, tokenizer, encoder, cached):
            return (.text(tokens: tokens, tokenizer: tokenizer, encoder: encoder, cached: cached), nil)
        case .encoding(let s): return (.encoding(seconds: s), nil)
        case let .step(image, index, total, sigma, seconds, h, w):
            return (.step(image: image, index: index, total: total, sigma: sigma, seconds: seconds,
                          latentGridHeight: h, latentGridWidth: w), nil)
        case .preview(let p):
            return (.preview(image: p.image, index: p.index, total: p.total, height: p.height, width: p.width, blob: id),
                    Data(p.rgba))
        case .decoding(let s): return (.decoding(seconds: s), nil)
        case let .image(index, r):
            return (.image(index: index, WorkerRender(r, blob: id)), WorkerImage.bytes(r.image))
        case let .unusable(image, stage): return (.unusable(image: image, stage: stage), nil)
        case .warning(let w): return (.warning(w), nil)
        case .held(let h): return (.held(h), nil)
        case .memory(let m): return (.memory(m), nil)
        }
    }

    /// The engine's event again; `blobs`: those received, by id.
    package func event(_ blobs: [UInt32: Data]) throws -> Engine.Event {
        func blob(_ id: UInt32) throws -> Data {
            guard let d = blobs[id] else {
                throw EngineError.internalFailure(component: "worker", detail: "the pixels of blob \(id) never came")
            }
            return d
        }
        switch self {
        case let .start(startStep, startSigma, evaluations, seeds, stages):
            return .start(Engine.Plan(denoising: DenoisingPlan(startStep: startStep, startSigma: startSigma, evaluations: evaluations),
                                      seeds: seeds, stages: stages))
        case let .stage(s, image): return .stage(s, image: image)
        case let .text(tokens, tokenizer, encoder, cached):
            return .text(tokens: tokens, tokenizer: tokenizer, encoder: encoder, cached: cached)
        case .encoding(let s): return .encoding(seconds: s)
        case let .step(image, index, total, sigma, seconds, h, w):
            return .step(image: image, index: index, total: total, sigma: sigma, seconds: seconds, latentGridHeight: h, latentGridWidth: w)
        case let .preview(image, index, total, height, width, id):
            let rgba = [UInt8](try blob(id))
            guard rgba.count == 4 * height * width else {
                throw EngineError.internalFailure(component: "worker", detail: "a preview of \(rgba.count) bytes for \(width)×\(height)")
            }
            return .preview(Engine.Preview(image: image, index: index, total: total, height: height, width: width, rgba: rgba))
        case .decoding(let s): return .decoding(seconds: s)
        case let .image(index, r): return .image(index: index, try r.render(try blob(r.image.blob)))
        case let .unusable(image, stage): return .unusable(image: image, stage: stage)
        case .warning(let w): return .warning(w)
        case .held(let h): return .held(h)
        case .memory(let m): return .memory(m)
        }
    }

    /// The blob it names, if any.
    package var blob: UInt32? {
        switch self {
        case .preview(_, _, _, _, _, let id): id
        case .image(_, let r): r.image.blob
        default: nil
        }
    }
}

/// **`Engine.Render` on the wire**, every field of it; the image by its blob.
package struct WorkerRender: Codable, Sendable {
    package struct Origin: Codable, Sendable {
        package var name: String
        package var sha256: String?
    }
    package var model: String, prompt: String
    package var seed: UInt64
    package var steps: Int
    package var loras: [LoRAEntry]
    package var image: WorkerImage
    package var evaluations: Int
    package var strength: Double?
    package var startStep: Int
    package var startSigma: Float
    package var timings: Engine.Timings
    package var footprints: Engine.Footprints
    package var tokens: Int
    package var detail: Detail
    package var solver: Solver
    package var variations: [Variation]
    package var modelName: String?
    package var isImportedModel: Bool
    package var loraOrigins: [Origin]

    init(_ r: Engine.Render, blob: UInt32) {
        model = r.model; prompt = r.prompt; seed = r.seed; steps = r.steps; loras = r.loras
        image = WorkerImage(blob: blob, width: r.image.width, height: r.image.height, alpha: r.image.rgba != nil)
        evaluations = r.evaluations; strength = r.strength; startStep = r.startStep; startSigma = r.startSigma
        timings = r.timings; footprints = r.footprints; tokens = r.tokens; detail = r.detail; solver = r.solver
        variations = r.variations; modelName = r.modelName; isImportedModel = r.isImportedModel
        loraOrigins = r.loraOrigins.map { Origin(name: $0.name, sha256: $0.sha256) }
    }

    func render(_ pixels: Data) throws -> Engine.Render {
        Engine.Render(model: model, prompt: prompt, seed: seed, steps: steps, loras: loras, image: try image.image(pixels),
                      evaluations: evaluations, strength: strength, startStep: startStep, startSigma: startSigma,
                      timings: timings, footprints: footprints, tokens: tokens, detail: detail, solver: solver,
                      variations: variations, modelName: modelName, isImportedModel: isImportedModel,
                      loraOrigins: loraOrigins.map { Engine.Render.LoRAOrigin(name: $0.name, sha256: $0.sha256) })
    }
}

// MARK: - The pipe

/// **One end of a pipe, written in order**: each send writes its blobs' pieces, then its message, under one lock — a
/// message never goes before the blobs it names, nor between another send's frames.
package final class WorkerPipe: @unchecked Sendable {
    private let fd: Int32
    private let lock = NSLock()
    private var broken = false

    package init(fd: Int32) { self.fd = fd }

    /// `false` once the other end is gone (`EPIPE`): nothing more is written.
    @discardableResult
    package func send<M: Encodable>(_ message: M, blobs: [(UInt32, Data)] = []) -> Bool {
        guard let json = try? WireJSON.encode(message) else { return false }
        var bytes = Data()
        for (id, data) in blobs {
            var offset = 0
            repeat {
                let end = min(offset + Outbox.pieceSize, data.count)
                var payload = Data(Frame.bigEndian(id))
                payload.append(data[(data.startIndex + offset)..<(data.startIndex + end)])
                bytes.append(Frame(kind: .piece, flags: end == data.count ? 1 : 0, payload: payload).bytes)
                offset = end
            } while offset < data.count
        }
        bytes.append(Frame(kind: .message, payload: json).bytes)
        return lock.withLock { () -> Bool in
            guard !broken else { return false }
            let ok = bytes.withUnsafeBytes { raw -> Bool in
                var done = 0
                while done < raw.count {
                    let n = Darwin.write(fd, raw.baseAddress! + done, raw.count - done)
                    if n < 0 { if errno == EINTR { continue }; return false }
                    done += n
                }
                return true
            }
            if !ok { broken = true }
            return ok
        }
    }
}

/// **The other end, read as it comes**: frames, blobs put back together, each message handed with the blobs it names.
package struct WorkerInbox {
    private var reader = FrameReader()
    /// Trusted, but bounded: an image of 4096² is 201 MB of floats; a few may wait for one call.
    private var assembler = BlobAssembler(maximumSize: 1 << 30, maximumOpen: 16)
    private var blobs: [UInt32: Data] = [:]

    package init() {}

    /// The messages the bytes complete, in order. A blob is kept until a message takes it (`take`).
    package mutating func receive(_ bytes: Data) throws -> [Data] {
        reader.append(bytes)
        var messages: [Data] = []
        while let f = try reader.next() {
            switch f.kind {
            case .message: messages.append(f.payload)
            case .piece: if let (id, data) = try assembler.receive(f) { blobs[id] = data }
            case .stream: throw FrameFailure.unknownKind(f.kind.rawValue)
            }
        }
        return messages
    }

    /// The blobs a message names, taken out: none outlives its message.
    package mutating func take(_ ids: [UInt32]) -> [UInt32: Data] {
        var taken: [UInt32: Data] = [:]
        for id in ids { taken[id] = blobs.removeValue(forKey: id) }
        return taken
    }
}
