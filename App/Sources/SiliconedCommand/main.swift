import Foundation
import Verification

// `siliconed` — the library (`models`, `install`, `verify`, `uninstall`) and every stage of every model checked against
// the reference implementation's goldens (`siliconed check list`). The models are the library's (`Library.processWide`).
setvbuf(stdout, nil, _IONBF, 0)
exit(Command.main(Array(CommandLine.arguments.dropFirst())))
