// **What `silicontrol measure` reads from the machine and from the app's process**, with what macOS
// ships and nothing else: `proc_pid_rusage` for the app's footprint and disk reads, `host_statistics64`
// for the swap-outs (the counter `vm_stat` prints), ImageIO for the pixels, IOKit's power-source API
// for the adapter. No tool is run, no Python, no Xcode.

import CryptoKit
import Darwin
import Foundation
import ImageIO
import IOKit.ps

package enum Probes {

    // MARK: The app's process

    /// The process at the other end of a connected socket (`LOCAL_PEERPID`): the app that answers, not
    /// one found by name — two instances (a test library beside the real one) cannot be confused.
    package static func peerPid(of fd: Int32) -> pid_t? {
        var pid: pid_t = 0
        var size = socklen_t(MemoryLayout<pid_t>.size)
        // SOL_LOCAL = 0, LOCAL_PEERPID = 2 (<sys/un.h>)
        return getsockopt(fd, 0, 2, &pid, &size) == 0 && pid > 0 ? pid : nil
    }

    package struct Usage: Sendable {
        package var footprint: UInt64      // ri_phys_footprint now
        package var diskRead: UInt64       // ri_diskio_bytesread since the process started
    }

    package static func usage(of pid: pid_t) -> Usage? {
        var r = rusage_info_v6()
        let status = withUnsafeMutablePointer(to: &r) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V6, $0) }
        }
        guard status == 0 else { return nil }
        return Usage(footprint: r.ri_phys_footprint, diskRead: r.ri_diskio_bytesread)
    }

    /// The `.app` a process runs from, and its Info.plist: the version of the app that is measured,
    /// whichever copy answered.
    package static func bundle(of pid: pid_t) -> (version: String, identifier: String)? {
        var buffer = [CChar](repeating: 0, count: 4096)
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return nil }
        var url = URL(fileURLWithPath: string(from: buffer))
        while url.path != "/" {
            if url.pathExtension == "app", let info = Bundle(path: url.path)?.infoDictionary {
                let short = info["CFBundleShortVersionString"] as? String ?? "?"
                let build = info["CFBundleVersion"] as? String
                return (build.map { $0 == short ? short : "\(short) (\($0))" } ?? short,
                        info["CFBundleIdentifier"] as? String ?? "dev.gwenn-ha.siliconed")
            }
            url.deleteLastPathComponent()
        }
        return nil
    }

    // MARK: Peak sampling

    /// `phys_footprint` of `pid` every `interval` seconds on a thread of its own, the maximum kept —
    /// the way the published peaks were taken (100 ms).
    package final class PeakSampler: @unchecked Sendable {
        private let lock = NSLock()
        private var peakValue: UInt64 = 0, running = true, diskReadValue: UInt64 = 0
        private let pid: pid_t, disk0: UInt64
        package init(pid: pid_t, interval: TimeInterval = 0.1) {
            self.pid = pid
            disk0 = Probes.usage(of: pid)?.diskRead ?? 0
            Thread.detachNewThread { [self] in
                while true {
                    lock.lock()
                    let go = running
                    lock.unlock()
                    guard go, let u = Probes.usage(of: pid) else { break }
                    lock.lock()
                    peakValue = max(peakValue, u.footprint)
                    diskReadValue = u.diskRead &- disk0
                    lock.unlock()
                    Thread.sleep(forTimeInterval: interval)
                }
            }
        }
        /// Stops the thread; the maximum, and the bytes the process read from disk since the start.
        package func stop() -> (peak: UInt64, diskRead: UInt64) {
            lock.lock(); running = false; lock.unlock()
            // One last read, so a spike in the final tenth of a second is not missed.
            let u = Probes.usage(of: pid)
            lock.lock(); defer { lock.unlock() }
            if let u { peakValue = max(peakValue, u.footprint); diskReadValue = u.diskRead &- disk0 }
            return (peakValue, diskReadValue)
        }
    }

    // MARK: The machine

    /// Pages swapped out since boot (the counter `vm_stat` prints as "Swapouts").
    package static func swapouts() -> UInt64 {
        var s = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &s) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        return status == KERN_SUCCESS ? s.swapouts : 0
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return string(from: buffer)
    }

    /// The text of a NUL-terminated C buffer.
    private static func string(from buffer: [CChar]) -> String {
        String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    package struct Host: Sendable, Equatable {
        package var chip: String, memoryGiB: Int, macOS: String
        package static func read() -> Host {
            var bytes: UInt64 = 0
            var size = MemoryLayout<UInt64>.size
            sysctlbyname("hw.memsize", &bytes, &size, nil, 0)
            let v = ProcessInfo.processInfo.operatingSystemVersion
            let build = sysctlString("kern.osversion") ?? "?"
            return Host(chip: sysctlString("machdep.cpu.brand_string") ?? sysctlString("hw.model") ?? "?",
                        memoryGiB: Int((Double(bytes) / 1_073_741_824).rounded()),
                        macOS: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion) (\(build))")
        }
    }

    /// "AC" or "battery": the power source in use now. A render on the battery is slower and its
    /// timing is not the published one.
    package static func powerSource() -> String {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return "?" }
        return type.hasPrefix("Battery") ? "battery" : type.hasPrefix("AC") ? "AC" : type
    }

    /// The system's thermal state and Low Power Mode, which both slow a series without saying so.
    package static func thermal() -> String {
        let p = ProcessInfo.processInfo
        let state = ["nominal", "fair", "serious", "critical"][min(3, p.thermalState.rawValue)]
        return p.isLowPowerModeEnabled ? state + "+lowpower" : state
    }

    // MARK: Pixels

    /// **MD5 of the decoded pixels**, 8-bit RGB row by row, no alpha, no metadata: what a PNG's
    /// bytes cannot say (the same picture encoded twice differs by its chunks), and what "same seed,
    /// same bits" means. Drawn in the image's own color space, so no conversion touches a value.
    package static func pixelDigest(png url: URL) -> String? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let w = image.width, h = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgbx = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgbx.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { return nil }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = rgbx[4 * i]; rgb[3 * i + 1] = rgbx[4 * i + 1]; rgb[3 * i + 2] = rgbx[4 * i + 2]
        }
        return Insecure.MD5.hash(data: rgb).map { String(format: "%02x", $0) }.joined()
    }
}
