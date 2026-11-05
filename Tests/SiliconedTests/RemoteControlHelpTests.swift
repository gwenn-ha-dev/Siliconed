import XCTest
@testable import SilicontrolHelp

/// **The remote control's help routing** — what `silicontrol` answers alone, without launching the
/// app, and what the app answers on its socket: the same function, so one test holds both.
final class RemoteControlHelpTests: XCTestCase {

    func testHelpRequestsAreAnswered() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            XCTAssertEqual(RemoteControlHelp.answer([], pages)?["help"] as? String, pages.overview)
            XCTAssertEqual(RemoteControlHelp.answer(["help"], pages)?["help"] as? String, pages.overview)
            XCTAssertEqual(RemoteControlHelp.answer(["--help"], pages)?["help"] as? String, pages.overview)
            XCTAssertEqual(RemoteControlHelp.answer(["help", "add"], pages)?["help"] as? String, pages.pages["add"])
            XCTAssertEqual(RemoteControlHelp.answer(["add", "--help"], pages)?["help"] as? String, pages.pages["add"])
            XCTAssertEqual(RemoteControlHelp.answer(["wait", "t3", "-h"], pages)?["help"] as? String, pages.pages["wait"])
            XCTAssertEqual(RemoteControlHelp.answer(["help", "examples"], pages)?["help"] as? String, pages.pages["recipes"])
            XCTAssertEqual(RemoteControlHelp.answer(["help", "add"], pages)?["ok"] as? Bool, true)
        }
    }

    /// Every topic of the reading order has a page in both languages, and `help all` holds them all.
    func testEveryTopicHasItsPages() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for topic in RemoteControlHelp.order { XCTAssertNotNil(pages.pages[topic], topic) }
            let all = RemoteControlHelp.answer(["help", "all"], pages)?["help"] as? String ?? ""
            for topic in RemoteControlHelp.order { XCTAssertTrue(all.contains(pages.pages[topic]!), topic) }
        }
    }

    /// An unknown topic is a usage failure with the topics as hint — `silicontrol` exits 2 on it.
    func testUnknownTopicIsAUsageFailure() {
        let a = RemoteControlHelp.answer(["help", "nope"], RemoteControlHelp.english)
        XCTAssertEqual(a?["ok"] as? Bool, false)
        XCTAssertEqual(a?["error"] as? String, "usage")
        XCTAssertEqual(a?["message"] as? String, "no help for \"nope\"")
        XCTAssertTrue((a?["hint"] as? String)?.hasPrefix("topics: contract,") == true)
        XCTAssertNil(a?["help"])
    }

    /// Anything else goes to the app, flags included: `--helper` is not `--help`.
    func testOtherCommandsAreNotHelp() {
        for argv in [["status"], ["add", "a woman posing in a library"], ["open"], ["add", "--helper"]] {
            XCTAssertNil(RemoteControlHelp.answer(argv, RemoteControlHelp.english), "\(argv)")
        }
    }

    /// **What the app refuses is documented**: no render during an installation — `add`,
    /// `grid` and `vary` list `busy` among their failures — and `install` names the three versions' flags.
    func testTheRefusalsOfC2bAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for topic in ["add", "grid", "vary"] {
                XCTAssertTrue(pages.pages[topic]?.contains("busy") == true, topic)
            }
            for flag in ["--standard", "--compact", "--light"] {
                XCTAssertTrue(pages.pages["install"]?.contains(flag) == true, flag)
            }
            XCTAssertTrue(pages.overview.contains("install ID [--standard|--compact|--light]"))
        }
    }

    /// **The Faster Rendering setting is documented**, in both languages: its command, its answer, its
    /// refusal, the series' option and `status`'s key.
    func testFasterRenderingIsDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            XCTAssertTrue(pages.overview.contains("set faster-rendering on|off"))
            XCTAssertTrue(pages.pages["set"]?.contains("\"previous\"") == true)
            XCTAssertTrue(pages.pages["set"]?.contains("busy") == true)
            XCTAssertTrue(pages.pages["measure"]?.contains("--faster-rendering on|off") == true)
            XCTAssertTrue(pages.pages["status"]?.contains("\"faster_rendering\"") == true)
            XCTAssertNotNil(RemoteControlHelp.answer(["set", "--help"], pages)?["help"] as? String)
        }
    }
}

/// **What `silicontrol` can do that the Prompt Lab and the Settings window do** is documented, in both languages: the commands in the
/// overview, a page for each new one, and every new refusal code in `help errors`.
final class RemoteControlWriterHelpTests: XCTestCase {
    func testTheLabAndTheSwitchesAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for line in ["install writer", "rewrite \"<", "describe <image|i7>", "set accelerators on|off", "set beta [OPTION on|off]"] {
                XCTAssertTrue(pages.overview.contains(line), line)
            }
            XCTAssertNotNil(RemoteControlHelp.answer(["rewrite", "--help"], pages)?["help"] as? String)
            XCTAssertNotNil(RemoteControlHelp.answer(["describe", "-h"], pages)?["help"] as? String)
            let errors = pages.pages["errors"] ?? ""
            for code in ["remote", "writer_absent", "writer_head", "writer_memory", "writer_preparing", "writer_busy",
                         "writer_failed", "prompt_too_long", "license_not_accepted"] {
                XCTAssertTrue(errors.contains(code), code)
            }
            for option in ["more-solvers", "faster-rendering", "size-scales", "negative-lora"] {
                XCTAssertTrue(pages.pages["set"]?.contains(option) == true, option)
            }
            XCTAssertTrue(pages.pages["install"]?.contains("install writer") == true)
            XCTAssertTrue(pages.pages["status"]?.contains("\"writer\"") == true)
            XCTAssertTrue(pages.pages["contract"]?.contains("loading, text") == true)
        }
    }

    /// **The Prompt Lab's buttons and the Settings window**, from the terminal: in the overview, on their pages, their new code in
    /// `help errors`.
    func testTheLabButtonsAndSettingsAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for line in ["lab rewrite · lab describe N [--wait]", "settings [", "settings lab [rewrite|describe] [--for ID]"] {
                XCTAssertTrue(pages.overview.contains(line), line)
            }
            for line in ["silicontrol lab rewrite [--wait]", "silicontrol lab describe <n> [--wait]", "empty_text", "\"writing\""] {
                XCTAssertTrue(pages.pages["lab"]?.contains(line) == true, line)
            }
            XCTAssertNotNil(RemoteControlHelp.answer(["settings", "--help"], pages)?["help"] as? String)
            for pane in ["general", "macs", "lab", "advanced"] {
                XCTAssertTrue(pages.pages["settings"]?.contains(pane) == true, pane)
            }
            XCTAssertTrue(pages.pages["errors"]?.contains("empty_text") == true)
        }
    }
}

/// **What `silicontrol` does of the window's other gestures is documented**, in both languages: the commands in the overview, a page for each
/// (and the aliases of the ones that share a page), the `--yes` guard where something is destroyed, and every new refusal code in `help errors`.
final class RemoteControlGestureHelpTests: XCTestCase {
    func testTheGesturesAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for line in ["remove i7", "undo [--redo]", "again <i7|", "reuse <i7|", "crop i7 --box X,Y,W,H | --ratio P:Q", "sheet g2",
                         "import FI", "install stop", "uninstall ID|lora", "macs [connect|disconnect", "compute N", "library [move", "--format F"] {
                XCTAssertTrue(pages.overview.contains(line), line)
            }
            for topic in ["remove", "undo", "again", "crop", "sheet", "import", "uninstall", "macs", "library"] {
                XCTAssertNotNil(pages.pages[topic], topic)
                XCTAssertNotNil(RemoteControlHelp.answer([topic, "--help"], pages)?["help"] as? String, topic)
            }
            // The alias commands read the page they share.
            for (command, topic) in [("reuse", "again"), ("compute", "macs")] {
                XCTAssertEqual(RemoteControlHelp.answer([command, "--help"], pages)?["help"] as? String, pages.pages[topic], command)
            }
            // What destroys says --yes, and what a dry run answers.
            for topic in ["uninstall", "library"] {
                XCTAssertTrue(pages.pages[topic]?.contains("--yes") == true, topic)
                XCTAssertTrue(pages.pages[topic]?.contains("dry_run") == true, topic)
            }
            let errors = pages.pages["errors"] ?? ""
            for code in ["nothing_to_undo", "crop", "cell", "import", "unknown_mac", "library", "link", "refused", "remote"] {
                XCTAssertTrue(errors.contains(code), code)
            }
            XCTAssertTrue(pages.pages["save"]?.contains("heic") == true)
            XCTAssertTrue(pages.pages["grid"]?.contains("g2c3") == true)
            XCTAssertTrue(pages.pages["install"]?.contains("install stop") == true)
        }
    }

    /// The commands the app routes (`GestureOrder.verbs`) each have a page, here or by alias.
    func testEveryGestureVerbHasAPage() {
        let verbs = ["remove", "undo", "again", "reuse", "crop", "sheet", "import", "uninstall", "macs", "compute", "library"]
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for verb in verbs { XCTAssertNotNil(RemoteControlHelp.text(for: verb, pages), verb) }
        }
    }
}
