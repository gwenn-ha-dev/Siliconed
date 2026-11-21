// **The remote control's socket address**, for both ends: the app that listens and `silicontrol` that connects.

import Foundation

package enum ControlSocket {
    /// The address of the socket file at `path`; `nil` when the path does not fit `sun_path`.
    package static func address(_ path: String) -> sockaddr_un? {
        var a = sockaddr_un()
        a.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard bytes.count < MemoryLayout.size(ofValue: a.sun_path) else { return nil }
        withUnsafeMutableBytes(of: &a.sun_path) { buffer in
            buffer.copyBytes(from: bytes)
            buffer[bytes.count] = 0
        }
        a.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        return a
    }

    /// A descriptor connected to the socket at `path`; `nil`: nobody answers there.
    package static func connect(_ path: String) -> Int32? {
        guard var a = address(path) else { return nil }
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        let r = withUnsafePointer(to: &a) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard r == 0 else { close(fd); return nil }
        return fd
    }
}
