import Foundation
import MLX

/// Coarse per-section profiler for the enhancer, gated by `REUSE_PROFILE=1`.
///
/// Each `measure` forces evaluation of its result, so measured sections do not overlap
/// on the GPU — absolute totals are inflated versus a single fused eval, but the relative
/// split is an accurate map of where time goes. Not thread-safe; the enhancer runs the
/// model on a single stream so that is fine.
public enum ReuseProfiler {
    public static let enabled =
        ProcessInfo.processInfo.environment["REUSE_PROFILE"] == "1"

    nonisolated(unsafe) private static var totals: [String: UInt64] = [:]
    nonisolated(unsafe) private static var order: [String] = []

    @inline(__always)
    public static func measure(_ name: String, _ body: () -> MLXArray) -> MLXArray {
        if !enabled { return body() }
        let start = DispatchTime.now().uptimeNanoseconds
        let result = body()
        eval(result)
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        if totals[name] == nil { order.append(name) }
        totals[name, default: 0] += elapsed
        return result
    }

    public static func reset() {
        totals.removeAll()
        order.removeAll()
    }

    public static func dump(_ header: String) {
        guard enabled else { return }
        var lines = ["[profile] \(header)"]
        let grand = totals.values.reduce(0, +)
        for name in order.sorted(by: { (totals[$0] ?? 0) > (totals[$1] ?? 0) }) {
            let ms = Double(totals[name] ?? 0) / 1_000_000.0
            let pct = grand > 0 ? Double(totals[name] ?? 0) / Double(grand) * 100 : 0
            lines.append(String(format: "  %-22@ %8.1f ms  %5.1f%%", name as NSString, ms, pct))
        }
        lines.append(String(format: "  %-22@ %8.1f ms", "TOTAL(measured)" as NSString, Double(grand) / 1_000_000.0))
        FileHandle.standardError.write((lines.joined(separator: "\n") + "\n").data(using: .utf8)!)
    }
}
