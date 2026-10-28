import Foundation

/// **What a lending Mac still holds for a client, said once at each join** — the history and the jobs
/// as the client must have them. A joined Mac is sent everything again (its `Seen` is new) and ignores
/// what it knows; this is the half that removal needs: an image deleted, or a job forgotten, while the
/// link was cut is not in these lists, and the client drops it.
package struct RemoteSync: Codable, Equatable, Sendable {
    /// The `i` ordinals of the client's images in the history.
    package var entries: [Int]
    /// The `t` ordinals of the client's jobs the lender knows: running, waiting, or ended in its session.
    /// Their ends are sent again after the images; a job of the client's that is in none of them is lost.
    package var jobs: [Int]
    package init(entries: [Int], jobs: [Int]) { self.entries = entries; self.jobs = jobs }
}

/// **The images a client removed, kept until it leaves** — in the lending Mac's memory only, never on
/// disk (the session is incognito): « Undo » asks them back by ordinal. Bounded: the oldest removed
/// is forgotten first.
package struct SessionTrash<Item> {
    package let capacity: Int
    private var items: [Int: Item] = [:]
    /// The ordinals, the oldest removed first.
    private var order: [Int] = []

    package init(capacity: Int = 64) { self.capacity = capacity }

    package var ordinals: [Int] { order }

    package mutating func put(_ ordinal: Int, _ item: Item) {
        order.removeAll { $0 == ordinal }
        order.append(ordinal)
        items[ordinal] = item
        while order.count > capacity { items[order.removeFirst()] = nil }
    }

    /// The items still held among `ordinals`, taken out of the trash.
    package mutating func take(_ ordinals: [Int]) -> [Int: Item] {
        var out: [Int: Item] = [:]
        for o in ordinals { if let i = items.removeValue(forKey: o) { out[o] = i } }
        order.removeAll { out[$0] != nil }
        return out
    }
}
