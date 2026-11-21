import Foundation

extension LatentSpace {
    package static let flux = LatentSpace(name: "VAE Flux", channels: 16, factor: 8, preview: .flux)
}

package struct FluxDecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }
}

/// The Flux VAE's encoder (img2img): the image at the render's format → `(mean − 0.1159) × 0.3611`.
package struct FluxEncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }
}
