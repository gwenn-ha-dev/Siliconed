import Foundation
import Testing
@testable import SiliCore

/// GELU against coreai-torch's own assets (`tools/coreai/references_elementwise.py`, as SiliCore recorded them at
/// ae45f2e), and the bf16 rounding the bf16 path reads its F32 parameters through.
private let geluReferences: [String: String] = [
    "gelu_bfloat16_3_5": "TUzvUg1NTElSMjIuMC4wZ2l0AAEZAwMHAwUJAQkHCw8TAzclBwElFxMfGxMHDw8TCwsLCw8PEwsLAQcXFwcCjVEHCwEBBQMBAAVBAQgyoCY/CQEBAQE9Cw0NXQsDDwsDERMPAxMDEQkDAxMLAQsDHQsDHxMPAyEDFScDBQUNFQU9AwEDAREETQUBUAEBBwQ9AwEFA1ABAwcEKQMFCwMDAQAFRgEFAwEDAQcEAQMDBgMBBQEA8xcLCwUZAwUnJSUpD2FpY29kZQBzZXJpYWxpemF0aW9uLmFzc2V0AGRhdGFmbG93LmdyYXBoX3YxAHByaW1pdGl2ZS5nZWx1X3YxAGRhdGFmbG93Lm91dHB1dF92MQAtAABhaWNvZGUubmFtZQB4AG1haW4AZ2VsdQAIIQcFAQsTDQEDFQMXAxkbAyM=",
    "gelu_float16_3_5": "TUzvUg1NTElSMjIuMC4wZ2l0AAEZAwMHAwUJAQkHCw8TAzclBwElFxMfGxMHDw8TCwsLCw8PEwsLAQcXFwcCjVEHCwEBBQMBAAVBAQgyoCY/CQEBAQE9Cw0NXQsDDwsDERMPAxMDEQkDAxMLAQsDHQsDHxMPAyEDFScDBQUNFQU9AwEDAQ8ETQUBUAEBBwQ9AwEFA1ABAwcEKQMFCwMDAQAFRgEFAwEDAQcEAQMDBgMBBQEA8xcLCwUZAwUnJSUpD2FpY29kZQBzZXJpYWxpemF0aW9uLmFzc2V0AGRhdGFmbG93LmdyYXBoX3YxAHByaW1pdGl2ZS5nZWx1X3YxAGRhdGFmbG93Lm91dHB1dF92MQAtAABhaWNvZGUubmFtZQB4AG1haW4AZ2VsdQAIIQcFAQsTDQEDFQMXAxkbAyM=",
    "geluerf_bfloat16_3_5": "TUzvUg1NTElSMjIuMC4wZ2l0AAEZAwMHAwUJAQkHCw8TAzclBwElFxMfGxMHDw8TCwsLCw8PEwsLAQcXFwcCjVEHCwEBBQMBAAVBAQgyoCY/CQEBAQE9Cw0NXQsDDwsDERMPAxMDEQkDAxMLAQsDHQsDHxMPAyEDFScBBQUNFQU9AwEDAREETQUBUAEBBwQ9AwEFA1ABAwcEKQMFCwMDAQAFRgEFAwEDAQcEAQMDBgMBBQEA8xcLCwUZAwUnJSUpD2FpY29kZQBzZXJpYWxpemF0aW9uLmFzc2V0AGRhdGFmbG93LmdyYXBoX3YxAHByaW1pdGl2ZS5nZWx1X3YxAGRhdGFmbG93Lm91dHB1dF92MQAtAABhaWNvZGUubmFtZQB4AG1haW4AZ2VsdQAIIQcFAQsTDQEDFQMXAxkbAyM=",
    "geluerf_float16_3_5": "TUzvUg1NTElSMjIuMC4wZ2l0AAEZAwMHAwUJAQkHCw8TAzclBwElFxMfGxMHDw8TCwsLCw8PEwsLAQcXFwcCjVEHCwEBBQMBAAVBAQgyoCY/CQEBAQE9Cw0NXQsDDwsDERMPAxMDEQkDAxMLAQsDHQsDHxMPAyEDFScBBQUNFQU9AwEDAQ8ETQUBUAEBBwQ9AwEFA1ABAwcEKQMFCwMDAQAFRgEFAwEDAQcEAQMDBgMBBQEA8xcLCwUZAwUnJSUpD2FpY29kZQBzZXJpYWxpemF0aW9uLmFzc2V0AGRhdGFmbG93LmdyYXBoX3YxAHByaW1pdGl2ZS5nZWx1X3YxAGRhdGFmbG93Lm91dHB1dF92MQAtAABhaWNvZGUubmFtZQB4AG1haW4AZ2VsdQAIIQcFAQsTDQEDFQMXAxkbAyM=",
]

@Test(arguments: geluReferences.keys.sorted())
func geluBytecodeEqualsTheReference(_ name: String) throws {
    let element: Aicode.Element = name.contains("bfloat16") ? .bf16 : .f16
    var m = Aicode.Module(arguments: [("x", Aicode.tensor([3, 5], element))])
    m.outputs = [("gelu", m.body.gelu(0, tanh: !name.hasPrefix("geluerf")))]
    #expect(m.bytecode() == [UInt8](Data(base64Encoded: geluReferences[name]!)!), "\(name)")
}

@Test func bfloat16RoundsToNearestEven() {
    #expect(BFloat16Bits.rounded(1) == 0x3F80)
    // 1 + 2⁻⁸ is half way between 1 and 1 + 2⁻⁷: to the even one, 1; 1 + 3·2⁻⁸ to 1 + 2⁻⁶.
    #expect(BFloat16Bits.rounded(1 + 0x1p-8) == 0x3F80)
    #expect(BFloat16Bits.rounded(1 + 3 * 0x1p-8) == 0x3F82)
    #expect(BFloat16Bits.widened(BFloat16Bits.rounded(-2.5)) == -2.5)
    #expect(BFloat16Bits.widened(BFloat16Bits.rounded(.infinity)) == .infinity)
    #expect(BFloat16Bits.widened(BFloat16Bits.rounded(.nan)).isNaN)
}
