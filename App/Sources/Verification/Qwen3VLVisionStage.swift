import Foundation
import Siliconed

// **Qwen3-VL's vision tower against the oracle** — the body of `siliconed check qwen-image-2.1 vision` (the stage is
// declared with the family's): the oracle's two reference pictures, each alone, from their `pixel_values`.

enum Qwen3VLVisionCheck {
    /// The first block's output, the last hidden state, the merged rows and the three deepstack rows of each picture,
    /// against transformers in bf16.
    static func judged(_ invocation: Invocation, model: String, map name: String) throws -> Report {
        let report = Report(title: "\(model) vision")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vision.safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-vision.safetensors", script: "tools/oracle/qwen-image-2.1.py vision-bf16")
        let path = try MapCheck.use(Library.processWide.map(name))
        let config = Qwen3VLVision.Config.eightB
        var pictures: [Qwen3VLImages.Pixels] = []
        for i in 0..<2 {
            guard let grid = goldens.int32("image_grid_thw_\(i)"), grid.count == 3, let pixels = goldens.float32("pixel_values_\(i)") else {
                throw Skip("\(goldens.path): picture \(i) missing")
            }
            pictures.append(Qwen3VLImages.Pixels(values: Array(pixels), gridHeight: Int(grid[1]), gridWidth: Int(grid[2])))
        }
        for (i, picture) in pictures.enumerated() {
            let out = try Qwen3VLVision.rows([picture], config: config, map: path, taps: Set(Qwen3VLVision.Tap.allCases))
            for tap in Qwen3VLVision.Tap.allCases {
                let name = "\(tap.rawValue)_\(i)"
                report.add(goldens, name, out[tap.rawValue], half: half.float32(name).map(Array.init), columns: config.width)
            }
            for (key, name) in [("merged", "vit_out_\(i)")] + config.deepstack.indices.map({ ("deepstack\($0)", "deepstack\($0)_\(i)") }) {
                report.add(goldens, name, out[key], half: half.float32(name).map(Array.init), columns: config.output)
            }
        }
        return report
    }
}
