import Foundation

/// **A publisher's base model, refused at the import where the family renders its Turbo.**
///
/// Z-Image, Krea 2, Anima and Qwen-Image-2.1 each publish a base model beside the Turbo Siliconed
/// renders, and each base is the Turbo's architecture: the same names and shapes, so
/// `Recipes.family(fromNames:)` and the forge (against `dit.json`) take it as the Turbo, and the family's
/// recipe — 8 steps on the Turbo's schedule, without CFG — renders it into a ruined image, without a
/// word (Z-Image's base asks 28 to 50 steps with CFG, Krea 2 Raw 52, Anima's base 30 to 50). Compared
/// header against header (3/11/2026):
///
///     Qwen-Image-2.1  diffusers shards: the very headers, offsets included; Comfy-Org's bf16 likewise;
///                     unsloth's GGUFs: every tensor's type and offset
///     Z-Image         Comfy-Org's bf16 and int8 convrot, unsloth's GGUFs: identical headers; the diffusers
///                     shards: the same 521 names and shapes, in bf16 where the Turbo's are fp32
///     Krea 2 (Raw)    Comfy-Org's bf16, int8 convrot and fp8 scaled, unsloth's diffusers shards,
///                     vantagewithai's GGUFs: identical headers
///     Anima           the same 685 names and shapes, under `net.` where the Turbo's are under
///                     `model.diffusion_model.`; Abiray's GGUFs: six tensors F16 where vanes430's are BF16
///
/// **Nothing in a header tells them apart**: `__metadata__` is empty or `{"format": "pt"}` on both sides,
/// unsloth's torchao files describe their quantization and nothing of the model, Z-Image's and Krea 2's
/// GGUFs carry the same keys as their Turbo's, and Qwen's base GGUFs none where the Turbo's carry two — an
/// absence, which no other publisher is bound to. **Only the bytes do**: every base checkpoint published
/// is listed here at a pinned revision with its size and sha256 (the LFS object the Hub lists), and a file
/// that is one of them, byte for byte, is `EngineError.importBaseModel`. Each is a base by its values too:
/// an unquantized tensor of one file per repository, read by `Range`, is the base's and not the Turbo's.
/// The test is `Adoption`'s: the size first, the hash — seconds on a large file — only when a size agrees.
/// A fine-tune, a merge or a re-quantization of a base is not here, and imports as before.
///
/// **Sizes alone do not decide**: Qwen's diffusers shards and bf16 files, Comfy-Org's Z-Image and Krea 2
/// files, Z-Image's fp32 shards of the first upload, most of Krea 2's GGUFs and seven of Z-Image's have
/// their Turbo's sizes to the byte.
///
/// MOSS-Image is not here: no MOSS-Image checkpoint is recognized (`Recipes.family(fromNames:)`), base or SFT.
///
/// Read by the import of a file (`ModelImport.recognize`, `importFile`) and by the Hub's offer, which
/// knows each file's size and sha256 from the listing and refuses before a byte is downloaded
/// (`HubImport.examine`). Never by `ModelImport.examine`, whose file is a header in a sparse file:
/// its hash is not the published one.
enum BaseModels {
    /// One published base checkpoint, of a family that renders its Turbo.
    struct Known {
        let family: Family
        let file: Installer.File
    }

    // Revisions read on 3/11/2026. The earlier ones hold checkpoints replaced since, still downloadable at
    // their commit: Qwen's before its 0919 release, Z-Image's first upload in fp32, unsloth's first quants.
    static let qwenImage21 = Installer.Repository(name: "Qwen/Qwen-Image-2.1", revision: "d26bb61231c349cf6b7896fa83353113880e1ba3")
    static let qwenImage21Before0919 = Installer.Repository(name: "Qwen/Qwen-Image-2.1", revision: "f52e6a529129fff6a994469561673ea45a84043d")
    static let qwenImage21ComfyBefore0919 = Installer.Repository(name: "Comfy-Org/Qwen-Image-2.1", revision: "8150226f50722886a275fa08e7b1fdf961732502")
    static let qwenImage21GGUF = Installer.Repository(name: "unsloth/Qwen-Image-2.1-GGUF", revision: "2c31ccd392b367a6637841a143813320a02dff55")
    static let zImage = Installer.Repository(name: "Tongyi-MAI/Z-Image", revision: "04cc4abb7c5069926f75c9bfde9ef43d49423021")
    static let zImageFP32 = Installer.Repository(name: "Tongyi-MAI/Z-Image", revision: "3c699857f341007c8d9d37690cc25301bd23514e")
    static let zImageComfy = Installer.Repository(name: "Comfy-Org/Z-Image", revision: "94d47a468ec820d29a5dc09d4642539d76d49542")
    static let zImageGGUF = Installer.Repository(name: "unsloth/Z-Image-GGUF", revision: "a84a5f4e7dcff6bc446c0a54a477a4a21aa800b0")
    static let zImageGGUFFirst = Installer.Repository(name: "unsloth/Z-Image-GGUF", revision: "8acd92073b207fc779b9c77e87934cbe92d4aa10")
    // Krea's own `krea/Krea-2-Raw` is gated (`Installer.krea2Comfy`): its ungated mirror by unsloth is listed.
    static let krea2Raw = Installer.Repository(name: "unsloth/Krea-2-Raw", revision: "62982f4e78c5f52ec16ec4aea2e8481950b73784")
    static let krea2RawGGUF = Installer.Repository(name: "vantagewithai/Krea-2-Raw-GGUF", revision: "c20e61051625c7ff0a16425f3a87e702486ea3a8")
    static let animaBaseGGUF = Installer.Repository(name: "Abiray/Anima-base-v1.0-GGUF", revision: "23bc46e07aa50a57b8c15a733bbcdf10889e62b0")

    /// Every base checkpoint published by the family's publisher and by the third parties of its Turbo's
    /// versions (Comfy-Org, unsloth, the GGUFs'): diffusers shards (one alone is not importable, but the
    /// first is recognized and would be refused later, less clearly), Comfy-Org's bf16 and 8-bit files,
    /// unsloth's torchao fp8 and int8, the GGUFs — those under 4 bits included, for this refusal says more
    /// than the one they would meet. Anima's: the base and its last preview (`preview3-base`); its Aesthetic
    /// versions are fine-tunes, not the base, and pass. unsloth's `raw.safetensors` is Comfy-Org's
    /// `krea2_raw_bf16` (one LFS object), listed once.
    static func published() -> [Known] {
        let comfy = Installer.qwenImage21Comfy, unsloth = Installer.qwenImage21Unsloth
        let files: [(Family, Installer.Repository, String, Int, String)] = [
            (.qwenImage21, qwenImage21, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 9968332504, "9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b"),
            (.qwenImage21, qwenImage21, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 4261951904, "3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa"),
            (.qwenImage21, qwenImage21Before0919, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 9968332504, "825e8d8526ca6135dad7b43b1e6d0ab3530bc135274a2e75f1e8510d515f1525"),
            (.qwenImage21, qwenImage21Before0919, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 4261951904, "60f53747c6ff3a58d665f63d16d1e232c6642f77bd098a32951b6f09d8840d73"),
            (.qwenImage21, comfy, "diffusion_models/qwen_image_2.1_bf16.safetensors", 14230280616, "89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc"),
            (.qwenImage21, comfy, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7256783064, "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d"),
            (.qwenImage21, qwenImage21ComfyBefore0919, "diffusion_models/qwen_image_2.1_bf16.safetensors", 14230284584, "b4c1ebf5167cbb568eddf9ddf0e1a25fa06888a7f057765f04fc9f9acc4bcc5e"),
            (.qwenImage21, qwenImage21ComfyBefore0919, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7256796872, "4ab3fd2516ac19231e76236ad78ac3d8d3b75e9dfedeca5a258d8c1ab60e9788"),
            (.qwenImage21, unsloth, "Qwen-Image-2.1-FP8.safetensors", 7122877560, "70e151cfbedd37de7f4a11bda8f58180965c85f4960584c3d10acff599fcd205"),
            (.qwenImage21, unsloth, "Qwen-Image-2.1-INT8-ConvRot.safetensors", 7258370288, "f5f30176b7314cdc0574fe80778c56ef6651a4ac2b89aa28bbc897818fb497f0"),
            (.qwenImage21, unsloth, "Qwen-Image-2.1-INT8.safetensors", 7258361376, "51ed9fe73c8780e43d91c2ff2f36bf609369a8af2761d617b9a10c83d6ea3c60"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-F16.gguf", 14230275808, "8aa239e5ddb28d4641942231396d7171da41ad305ea401dd92a7d6410bde798d"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q2_K.gguf", 2466137824, "053e2cf11059e9f5091c03853025b18bc64d18fff4426b1a17da09ed29cc09c9"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q3_K_M.gguf", 3168290528, "f688e47c32c033aa8d059c6b7e27d668b2cef0f1dd3d667f97f544954d541238"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q3_K_S.gguf", 2724742880, "e71af4723442d004b469425ddbdbbf5e4f5f6068c2544cd82d9bb49791b54be3"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q3_K_XL.gguf", 3612493536, "ef42f3a1b0ccf30be0a068341d383beeea86173e6aa1aa1a33749b337e55e0a9"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q4_K_M.gguf", 4199565024, "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q4_K_S.gguf", 3906356960, "328a2b2203ceb45a7f14515fb902c5ec28e9df901d43f49020b4fd251234587f"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q5_K_M.gguf", 5390223072, "4b53321654dd3bf0aa8dd6bb821fdcb1c9053ca735070e359a23cbb9f97268ce"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q5_K_S.gguf", 4501948128, "ba9399e07005c929d2cba313d73f0658e7893219825f29dc609130b75803ea75"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q6_K.gguf", 6271551200, "2ead1ee665e8ebc5fec517d287a167f5c6a7e985acbf3b1b93143437a04616ac"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q6_K_XL.gguf", 6718506720, "9410ad28f20018e7f2f300a06eed7e48db144286d02f0a6d4208d4dbd266691b"),
            (.qwenImage21, qwenImage21GGUF, "qwen-image-2.1-Q8_0.gguf", 7640860384, "c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5"),
            (.zImage, zImage, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 9973727144, "ecd5df7768856215812af84d5cc785ba9d84c6d8c84e6939c7ccf30a2e0d2425"),
            (.zImage, zImage, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 2336146728, "e2d5fb75ca504b2d669a33af9380b68e0c1632ecddbfca85f82631958a4b81b1"),
            (.zImage, zImageFP32, "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 9973693184, "c0a0fdf1134e5633e319c916aa613222bbb344e4e4e07ee88c39d0a09e9915e5"),
            (.zImage, zImageFP32, "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 9973714824, "66e299554a3f0a636db60ad71250cb23b67e8b09c42298697b01b805d7fb473c"),
            (.zImage, zImageFP32, "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 4672282880, "881fd7b9a6b21279b1895a0716303ec6ea2fb1821366fc4dd4c8992a7911a42c"),
            (.zImage, zImageComfy, "split_files/diffusion_models/z_image_bf16.safetensors", 12309866400, "996a67d3ff666946b1c25cbc16d1b1918b6cc0ac166309e23fe3b3d830263dee"),
            (.zImage, zImageComfy, "split_files/diffusion_models/z_image_int8_convrot.safetensors", 6201001296, "eb2b470694fbd9778f15efd309000ea0091312c018c10a352c120b1da9bf4279"),
            (.zImage, zImageGGUF, "z-image-BF16.gguf", 12311939136, "82eebf65df40cb09cf29094e30b3cdf7b628e873b7fe5f32023ddb6f0188490c"),
            (.zImage, zImageGGUF, "z-image-F16.gguf", 12311939136, "9e1d3fd6f88e00eda20932b860e2369dbe26b412f196dbf7cf63e4665fe61965"),
            (.zImage, zImageGGUF, "z-image-Q2_K.gguf", 2311657536, "e8dd7a5d36bc85c986f666d8e381b453a23b2af07fb611619d8e8b655b4187f3"),
            (.zImage, zImageGGUF, "z-image-Q3_K_L.gguf", 4605642816, "6f59c9176e7207bd0b6c8f0838b7e7552282d1afeb8a8b80e19e08657e8cf5f9"),
            (.zImage, zImageGGUF, "z-image-Q3_K_M.gguf", 3334564416, "77f1e77261cbc1e2a2ed2391f3303cfaffe1f33c2a68edafd32244bfd0a54c55"),
            (.zImage, zImageGGUF, "z-image-Q3_K_S.gguf", 4361818176, "97b718afb754c16062e9d8ec9e3e53040be7eb3c28295c195e5b9058a821d3da"),
            (.zImage, zImageGGUF, "z-image-Q4_0.gguf", 4585244736, "b01b5433eed8d8264824a6eda366166952c8ea4c13ce3c107c88da8909327320"),
            (.zImage, zImageGGUF, "z-image-Q4_1.gguf", 4850665536, "2f8ac42be364d155e2e7e224456cc7b23e6f55a41c08559ae38a97b2e7ef5934"),
            (.zImage, zImageGGUF, "z-image-Q4_K_M.gguf", 3850115136, "e7fe0cd70917d8f266df936672210b5397bca8092a22608cf2c1fdc266f78a8c"),
            (.zImage, zImageGGUF, "z-image-Q4_K_S.gguf", 4789148736, "d400d05ee598c9f556b89d01824dedb019ea5f7b8bd2f7fa0be5212e3cc192ab"),
            (.zImage, zImageGGUF, "z-image-Q5_0.gguf", 5263542336, "f33b82eec2049549698403dcb36bdfea6137982d864ffccf88d9b5fb367777dd"),
            (.zImage, zImageGGUF, "z-image-Q5_1.gguf", 5528963136, "c44de84e2b4e5b119fab882c8f588ba852561a6d952b239c50719fe7c825a115"),
            (.zImage, zImageGGUF, "z-image-Q5_K_M.gguf", 5576771136, "632265e4765fbb6cae7a3f95de3d90802ae74c5511f61855e42ec2672e07d3df"),
            (.zImage, zImageGGUF, "z-image-Q5_K_S.gguf", 5291274816, "edaac89f9bd5e0ad5006ba5559c5a158aa4700a285c6b14c1e824525514dc4d3"),
            (.zImage, zImageGGUF, "z-image-Q6_K.gguf", 6102927936, "0b9be4cfcd23e57ef1b2a1802d0a1633c7ddfff09abe394c2bf35f5fd52cc6b5"),
            (.zImage, zImageGGUF, "z-image-Q8_0.gguf", 7224707136, "a51d4e36f84caa972e0eff65e8bd2961add14374d8a7038a51eefea2fe45f2c9"),
            (.zImage, zImageGGUFFirst, "z-image-Q2_K.gguf", 4013115456, "9c1857629988bcd6ff3468e5b2fe6c770aa26591f984bd91eecad18f36c77e45"),
            (.zImage, zImageGGUFFirst, "z-image-Q3_K_L.gguf", 4604183616, "0912f25ee35775138166e0ddf3f6ec9fca8562c5cb4cc2b346652f238ff1038d"),
            (.zImage, zImageGGUFFirst, "z-image-Q3_K_M.gguf", 4559946816, "e0382d4b1affe9e552392aa9c53a20c2d661b4ddd7f8c56f1f34626c2538368c"),
            (.zImage, zImageGGUFFirst, "z-image-Q3_K_S.gguf", 4360190016, "04be2a299f4bb62df12fae37a3e93ebbfb0ebc899f493e3959d93b3c2de4591e"),
            (.zImage, zImageGGUFFirst, "z-image-Q4_0.gguf", 4585244736, "4c5cfc02e6007ae1f0b0d690f68bda452f51f7b8ab6d9f500f8c5b829fea4377"),
            (.zImage, zImageGGUFFirst, "z-image-Q4_1.gguf", 4850665536, "cb8afc552ab71354b5d3cf6592c14bc378857b987df324e6c638bb6ca67d2d99"),
            (.zImage, zImageGGUFFirst, "z-image-Q4_K_M.gguf", 5066995776, "a62b929f76553b21f68894e9ed34d24b7fb67fb59b5689fa06981865986cce40"),
            (.zImage, zImageGGUFFirst, "z-image-Q4_K_S.gguf", 4787443776, "1a4329507c97af2ac6ea81d69e89dc0a57eeb6117dcddbc6d1b3bd082f0ecde0"),
            (.zImage, zImageGGUFFirst, "z-image-Q5_0.gguf", 5263542336, "b1894d2902090e7a9bce21446125a5313ab0955f722a862115fafecc02fbadbf"),
            (.zImage, zImageGGUFFirst, "z-image-Q5_1.gguf", 5528963136, "9f8e033c14c54c4a3943c1ac67958aaf56f777fb4c3556ee77d61e172b029e2e"),
            (.zImage, zImageGGUFFirst, "z-image-Q5_K_M.gguf", 5578099776, "266a5e7d8e0ab2494cb202d7bd8fcfd79b2161b0678252f3e854e992844811c6"),
            (.zImage, zImageGGUFFirst, "z-image-Q5_K_S.gguf", 5289562176, "67b2ebf5e743a4d318644773602ea7cf0ca90bb42b6ff3254c9f13e301bfc8d0"),
            (.zImage, zImageGGUFFirst, "z-image-Q6_K.gguf", 6101921856, "cbe468f6095fb3fe8089c015e6c10c6f8416fdd05407865d431fe3493bed75c0"),
            (.krea2, Installer.krea2Comfy, "diffusion_models/krea2_raw_bf16.safetensors", 26283332608, "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7"),
            (.krea2, Installer.krea2Comfy, "diffusion_models/krea2_raw_int8_convrot.safetensors", 13492686496, "5585a4a38c4bcfb6fde2d480a4aa6edf7f665721ebde56d30662c35a45f5fa5c"),
            (.krea2, Installer.krea2Comfy, "diffusion_models/krea2_raw_fp8_scaled.safetensors", 13141730784, "48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61"),
            (.krea2, krea2Raw, "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 9970652128, "aeed04b3fd933ff35ffb898548a585a32f2146ee6681700e2b50e4478d52d58a"),
            (.krea2, krea2Raw, "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 9829444456, "e606797c9fbfeb820b9a0222823915ac1459d11a27caa84d53aba150c85f2b26"),
            (.krea2, krea2Raw, "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 6483240784, "9300cac5c4cdbc33bbccadf176c0e5a8314b6a4aabefae7a2781782d50d78e7d"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q2_K.gguf", 4889589024, "d7275969ccb0d3f253226ca9c478ebee227cb701d712913cc90d80323a400d1e"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q3_K_M.gguf", 6014825760, "00791e245167376acc0a58ffdb4ca9c0065f1838c6045aae07b89cdc40ba931e"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q3_K_S.gguf", 6014825760, "7fd0a577a81310a277f0404d96d072e9b1dccc44e1fb042ac97ac827dee268b2"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q4_0.gguf", 7486289184, "3b0fb190674494d702c9c387273cd49b2059d3e7b772676c563a34b8eeb43616"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q4_1.gguf", 8178742560, "5c73729f144c50c6021fff40b5e9a13c36238f18cfc547ae9ca3b012cd8139e9"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q4_K_M.gguf", 7486289184, "f6ad9fcdfb45423612b1464ed9a324e94f47027c63ce0b8d73b02a92deb40833"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q4_K_S.gguf", 7486289184, "dfb4ad182caa3dd3c00bbbdf465014dcf1b14732455d1a961f322536f3a4b106"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q5_0.gguf", 8871195936, "07c302cbd013c8b70fbbd5e67d4f6606e6a086b5c44287a014d3e2640fae3ecb"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q5_1.gguf", 9676559648, "3a7e03e4cfb6fe71eb833a9a888490984cb1f3c9dbabb8c7ccd284f577eb03d0"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q5_K_M.gguf", 8871195936, "0234dfb5220193dff31a98ae151ad09c812da5b390bea346b2a0b5c7cae1b8e2"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q5_K_S.gguf", 8871195936, "e4867c23aa2880befb17c36ade26c353e46fd23a0ef44de279c4afe6c3cfdf50"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q6_K.gguf", 10583638304, "af376b6a9e1a82b88f58eae26211001caa5e9468b84e958f2c6264690a406037"),
            (.krea2, krea2RawGGUF, "krea2_raw-Q8_0.gguf", 13705958688, "4540af78d6dfe7bb5a468f4ba4f5c8422bf7191055c093161d6d03677e98c7de"),
            (.anima, Installer.anima, "split_files/diffusion_models/anima-base-v1.0.safetensors", 4182218328, "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"),
            (.anima, Installer.anima, "split_files/diffusion_models/anima-preview3-base.safetensors", 4182218360, "14fffe8ad5116cd73b9a4696f6a89d7e5f6efdd24b2e4785603aa891a9b2295b"),
            (.anima, Installer.animaBase, "transformer/diffusion_pytorch_model.safetensors", 3912877104, "81dc5915fc3c3b170684534ca11ed6c535ddea5658793a801fac53b8d5281065"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q3_K_M.gguf", 1022137472, "700588908c7279c65b6186646bb5931cf9dc714bb215cf2515465ef60ead8afb"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q4_K_M.gguf", 1384407168, "bd56586c0d40bddc52395de1a2cab5dab2344d13012b0a8677fdca5905a4bcdc"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q4_K_S.gguf", 1269080192, "dfdc31a53edae35e1485c3f647214bc78ffb4fe3855f35a8946dd562fb0ccc3b"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q5_K_M.gguf", 1555636352, "8d6da4d6e5771e15aa10fd79f00eccfbcb48f6f414ff40794756488dd2d60a42"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q6_K.gguf", 1737567360, "712fa483e7b61688d7a6957780633d48cd98b7890fc8a92bfb8e42507667ae16"),
            (.anima, animaBaseGGUF, "anima-base-v1.0-Q8_0.gguf", 2239471744, "462b913268313ea64924d84de0889867fcb24a129d0940e33d4648d4d6286fa9"),
        ]
        return files.map { Known(family: $0.0, file: Installer.File(repository: $0.1, path: $0.2, destination: nil, sha256: $0.4, size: $0.3)) }
    }

    /// **A file listed by the Hub**, judged on the size and sha256 the listing announces: nothing is hashed.
    static func known(size: Int, sha256: String, among table: [Known] = published()) -> Known? {
        table.first { $0.file.size == size && $0.file.sha256 == sha256.lowercased() }
    }

    /// **A file on disk**: the size, then the sha256 only when a size agrees (`Adoption.matching`).
    static func known(_ path: String, among table: [Known] = published(),
                      hash: (String) throws -> String = Hashes.shared.sha256) throws -> Known? {
        try Adoption.matching(path, among: table, file: \.file, hash: hash).first
    }

    /// **The refusal**: `importBaseModel` when `path` is a published base checkpoint, nothing otherwise.
    static func refuse(_ path: String, among table: [Known] = published(),
                       hash: (String) throws -> String = Hashes.shared.sha256) throws {
        guard let k = try known(path, among: table, hash: hash) else { return }
        throw EngineError.importBaseModel(file: (path as NSString).lastPathComponent, model: ModelCard.of(k.family).id)
    }

    /// **One hash per file, not two**: the app asks `recognize` before `importFile`, on the same file, and a
    /// Turbo checkpoint of the base's size — a fine-tune saved like Comfy-Org's bf16 — would be hashed
    /// twice. Kept while the file keeps its size, its modification date and its inode.
    final class Hashes: @unchecked Sendable {
        static let shared = Hashes()
        private let lock = NSLock()
        private var known: [String: String] = [:]

        func sha256(_ path: String) throws -> String {
            // The stamps of what the hash reads, through a link (`stat`): the link's own would not change with its file.
            let key = "\(path)|\(MemoryNeed.InstalledFiles(paths: [path]).stamps)"
            if let sha = lock.withLock({ known[key] }) { return sha }
            let sha = try Downloader.sha256(path)
            lock.withLock {
                if known.count >= 16 { known.removeAll() }
                known[key] = sha
            }
            return sha
        }
    }
}
