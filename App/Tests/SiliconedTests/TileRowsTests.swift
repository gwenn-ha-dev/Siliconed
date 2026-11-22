import CoreGraphics
import XCTest
@testable import Siliconed

/// **The Prompt Lab's image box, by whole rows** (`TileRows`): the heights its grip stops at, from the tiles' frames; the
/// box's height for the rows kept; the rows a drag asks for. Pure: what `LabImages` lays out.
final class TileRowsTests: XCTestCase {
    /// Two rows of tiles of different heights, centered on their row, in any order: a row's bottom is its tallest tile's.
    private let frames = [
        CGRect(x: 200, y: 10, width: 80, height: 92),     // row 1, middle 56
        CGRect(x: 0, y: 0, width: 150, height: 112),      // row 1, middle 56, the tallest
        CGRect(x: 0, y: 126, width: 84, height: 112),     // row 2, middle 182
        CGRect(x: 100, y: 126.5, width: 84, height: 111), // row 2, its middle half a point off
    ]

    func testARowStopsUnderItsTallestTile() {
        XCTAssertEqual(TileRows.stops(frames, padding: 2), [114, 240])
        XCTAssertEqual(TileRows.stops([], padding: 2), [])
    }

    /// The rows kept, or all there are when fewer, one at least; one tile's height before the tiles are laid out.
    func testTheBoxShowsTheRowsKept() {
        let stops = TileRows.stops(frames, padding: 2)
        XCTAssertEqual(TileRows.height(stops, rows: 1, empty: 116), 114)
        XCTAssertEqual(TileRows.height(stops, rows: 2, empty: 116), 240)
        XCTAssertEqual(TileRows.height(stops, rows: 10, empty: 116), 240)
        XCTAssertEqual(TileRows.height(stops, rows: 0, empty: 116), 114)
        XCTAssertEqual(TileRows.height([], rows: 2, empty: 116), 116)
    }

    /// The grip goes to the row nearest the hand, among the first `limit`.
    func testTheGripStopsOnTheNearestRow() {
        let stops: [CGFloat] = [114, 240, 366]
        XCTAssertEqual(TileRows.rows(nearest: 150, in: stops, limit: 10), 1)
        XCTAssertEqual(TileRows.rows(nearest: 200, in: stops, limit: 10), 2)
        XCTAssertEqual(TileRows.rows(nearest: 1000, in: stops, limit: 10), 3)
        XCTAssertEqual(TileRows.rows(nearest: 1000, in: stops, limit: 2), 2)
        XCTAssertNil(TileRows.rows(nearest: 100, in: [], limit: 10))
    }
}
