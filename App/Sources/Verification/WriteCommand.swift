import Foundation
import Siliconed

/// **`siliconed write`** — the prompt writer outside the app (`PromptWriter`): a text rewritten, or a picture
/// described, for a family, under the instruction the Prompt Lab sends for it; the text printed as it comes.
enum WriteCommand {
    static let usage = """
        siliconed write rewrite "<text>" [--for <family>] [--system "<instruction>"] [--tokens N]
        siliconed write describe <image> [--for <family>] [--system "<instruction>"] [--ask "<text>"] [--tokens N]
        --for      whose style the instruction asks for (default: the family whose encoder writes)
        --system   an instruction in place of that one
        --ask      describe: a text in the user turn, after the picture
        --tokens   the most tokens the answer may take (default: the product's, \(PromptWriter.Limits().newTokens))
        """

    static func run(_ arguments: [String]) -> Int32 {
        var rest = arguments
        func option(_ name: String) -> String? {
            guard let i = rest.firstIndex(of: name), i + 1 < rest.count else { return nil }
            defer { rest.removeSubrange(i...(i + 1)) }
            return rest[i + 1]
        }
        let system = option("--system"), ask = option("--ask") ?? "", target = option("--for")
        var limits = PromptWriter.Limits()
        if let tokens = option("--tokens") {
            guard let n = Int(tokens), n > 0 else {
                FileHandle.standardError.write("✗ --tokens \(tokens): a count of tokens, 1 or more\n\(usage)\n".data(using: .utf8)!)
                return 64
            }
            limits.newTokens = n
        }
        guard rest.count == 2, let gesture = PromptWriterInstructions.Gesture(rawValue: rest[0]) else {
            FileHandle.standardError.write((usage + "\n").data(using: .utf8)!)
            return 64
        }
        do {
            let library = Library.processWide
            guard let writer = PromptWriter(library: library, limits: limits), let own = PromptWriter.family(in: library) else {
                throw PromptWriter.Failure.notInstalled
            }
            let family: Family
            if let target {
                guard let named = Family(rawValue: target) else {
                    throw ProgramError("unknown family \(target): \(Family.allCases.map(\.rawValue).joined(separator: ", "))")
                }
                family = named
            } else {
                family = own
            }
            let instruction = system ?? PromptWriterInstructions.instruction(for: family, gesture)
            let request: PromptWriter.Request
            switch gesture {
            case .rewrite: request = .rewrite(text: rest[1], instruction: instruction)
            case .describe:
                let image = try PromptWriterImage.rgb8(from: Data(contentsOf: URL(fileURLWithPath: rest[1])))
                request = .describe(image: image, instruction: instruction, ask: ask)
            }
            let line = try Run(writer.write(request)).wait()
            print(line)
            return 0
        } catch {
            FileHandle.standardError.write("✗ \(error)\n".data(using: .utf8)!)
            return 1
        }
    }

    /// One run, read to its end from a command that has no run loop: the text printed as it grows, then the line that
    /// times it.
    private final class Run: @unchecked Sendable {
        private let stream: AsyncThrowingStream<PromptWriter.Event, Error>
        private var result: Result<String, Error> = .success("")

        init(_ stream: AsyncThrowingStream<PromptWriter.Event, Error>) { self.stream = stream }

        func wait() throws -> String {
            let done = DispatchSemaphore(value: 0)
            Task.detached { [self] in
                do { result = .success(try await read()) } catch { result = .failure(error) }
                done.signal()
            }
            done.wait()
            return try result.get()
        }

        private func read() async throws -> String {
            let started = Date()
            var first: Date?, printed = "", tokens = 0
            for try await event in stream {
                switch event {
                case .loading(let fraction):
                    if fraction >= 1 { first = Date() }
                case .text(let text):
                    // Cumulative, and a loop cut shrinks it: only what extends the printed text is printed.
                    if text.hasPrefix(printed) {
                        print(text.dropFirst(printed.count), terminator: "")
                        fflush(stdout)
                        printed = text
                    }
                case .finished(_, let count): tokens = count
                case .cancelled: break
                }
            }
            print()
            let end = Date(), wait = first.map { $0.timeIntervalSince(started) } ?? 0
            let decoding = first.map { end.timeIntervalSince($0) } ?? 0
            return String(format: "— %d tokens written in %.1f s: first token after %.1f, then %.1f tokens/s",
                          tokens, end.timeIntervalSince(started), wait, decoding > 0 ? Double(max(tokens - 1, 0)) / decoding : 0)
        }
    }
}
