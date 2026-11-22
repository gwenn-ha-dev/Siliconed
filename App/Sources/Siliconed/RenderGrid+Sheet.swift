// A grid's sheet: its cells, labels and title laid out for renders of a size.

import CoreGraphics
import Foundation

extension RenderGrid {
    /// The sheet of this grid, for renders of `width × height`.
    package func sheet(imageWidth: Int, imageHeight: Int) -> Sheet {
        Sheet(columns: columns, rows: rows, imageWidth: imageWidth, imageHeight: imageHeight, rowLabels: y != nil)
    }

    /// **Where everything goes on the sheet**, in pixels, the origin at the top left:
    ///
    ///     ┌──────────────────────────────────────────────┐
    ///     │ title (2 lines) · what the cells share       │  titleHeight
    ///     │            │ label x0 │ label x1 │ …         │  headerHeight
    ///     │ label y0   │  cell    │  cell    │           │
    ///     │ label y1   │  cell    │  cell    │           │
    ///     └──────────────────────────────────────────────┘
    ///       margin, rowLabelWidth (0 without Y), gap between cells
    ///
    /// Pure arithmetic: an app draws the images and the text in these rectangles.
    package struct Sheet: Equatable, Sendable {
        /// A cell's long side at most: a 1024² render is reduced to 512 on the sheet.
        package static let maxCellSide = 512
        /// All the cells together at most (pixels): the sheet of 64 cells stays near 14 Mpx.
        package static let maxCellsArea = 12_000_000
        /// A cell's long side at least, however many cells.
        package static let minCellSide = 128

        package let columns: Int, rows: Int
        package let cellWidth: Int, cellHeight: Int
        /// The font sizes, in pixels: they follow the cells, so that a label reads at the cells' scale.
        package let labelSize: Int, titleSize: Int
        package let margin: Int, gap: Int
        package let titleHeight: Int, headerHeight: Int, rowLabelWidth: Int
        package let width: Int, height: Int

        package init(columns: Int, rows: Int, imageWidth: Int, imageHeight: Int, rowLabels: Bool) {
            let columns = max(1, columns), rows = max(1, rows)
            let long = max(1, imageWidth, imageHeight), short = max(1, min(imageWidth, imageHeight))
            // The long side such that all the cells together stay within `maxCellsArea`.
            let byArea = Int((Double(Self.maxCellsArea) / Double(columns * rows) * Double(long) / Double(short)).squareRoot())
            let side = max(Self.minCellSide, min(Self.maxCellSide, long, byArea) / 8 * 8)
            cellWidth = max(1, Int((Double(imageWidth) * Double(side) / Double(long)).rounded()))
            cellHeight = max(1, Int((Double(imageHeight) * Double(side) / Double(long)).rounded()))
            self.columns = columns
            self.rows = rows
            labelSize = max(13, Int((Double(side) / 24).rounded()))
            titleSize = Int((Double(labelSize) * 1.3).rounded())
            margin = max(16, labelSize)
            gap = max(6, labelSize / 3)
            // Two lines of title, one line of what the cells share, and air under them.
            titleHeight = Int((Double(titleSize) * 2.7 + Double(labelSize) * 1.6).rounded()) + gap
            headerHeight = labelSize * 2
            rowLabelWidth = rowLabels ? max(140, labelSize * 9) : 0
            width = 2 * margin + (rowLabels ? rowLabelWidth + gap : 0) + columns * cellWidth + (columns - 1) * gap
            height = 2 * margin + titleHeight + headerHeight + rows * cellHeight + (rows - 1) * gap
        }

        private var left: Int { margin + (rowLabelWidth > 0 ? rowLabelWidth + gap : 0) }
        private var top: Int { margin + titleHeight + headerHeight }

        package func cell(column: Int, row: Int) -> CGRect {
            CGRect(x: left + column * (cellWidth + gap), y: top + row * (cellHeight + gap),
                   width: cellWidth, height: cellHeight)
        }

        /// Above its column, the width of a cell.
        package func columnLabel(_ column: Int) -> CGRect {
            CGRect(x: left + column * (cellWidth + gap), y: margin + titleHeight, width: cellWidth, height: headerHeight)
        }

        /// Left of its row, the height of a cell; empty without a Y axis.
        package func rowLabel(_ row: Int) -> CGRect {
            CGRect(x: margin, y: top + row * (cellHeight + gap), width: rowLabelWidth, height: cellHeight)
        }

        /// The title band, the sheet's width inside the margins.
        package var title: CGRect {
            CGRect(x: margin, y: margin, width: width - 2 * margin, height: titleHeight - gap)
        }
    }
}
