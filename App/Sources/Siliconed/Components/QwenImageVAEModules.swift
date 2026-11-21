import Foundation

// The Qwen-Image VAE as a chain reads it: its latent space, its encoding and decoding modules.

extension LatentSpace {
    package static let qwenImage = LatentSpace(name: "VAE Qwen-Image", channels: 16, factor: 8, preview: .qwenImage)
}

/// The Qwen-Image VAE's encoder (img2img): the
/// image at the render's format → `(mean − mean_c) / std_c`.
package struct QwenImageEncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage }
    package var name: String { "VAE Qwen-Image" }
    package init(path: String) { self.path = path }
}

package struct QwenImageDecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage }
    package var name: String { "VAE Qwen-Image" }
    package init(path: String) { self.path = path }
}
