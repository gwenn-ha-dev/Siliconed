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

    /// **After a lone `--`, `--help` and `-h` are words** — a prompt's —; before it, they ask for the command's page.
    func testHelpIsAskedBeforeTheLoneSeparatorOnly() {
        let pages = RemoteControlHelp.english
        for argv in [["add", "--", "-h"], ["add", "--", "--help"], ["add", "--wait", "--", "a cat", "-h"], ["rewrite", "--", "--help me"]] {
            XCTAssertNil(RemoteControlHelp.answer(argv, pages), "\(argv)")
        }
        XCTAssertEqual(RemoteControlHelp.answer(["add", "-h", "--", "a cat"], pages)?["help"] as? String, pages.pages["add"])
        XCTAssertEqual(RemoteControlHelp.answer(["add", "a cat", "--help"], pages)?["help"] as? String, pages.pages["add"])
    }

    /// **What the app refuses is documented**: no render during an installation — `add`,
    /// `grid` and `vary` list `busy` among their failures — and `install` names the versions' flags.
    func testTheRefusalsOfC2bAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for topic in ["add", "grid", "vary"] {
                XCTAssertTrue(pages.pages[topic]?.contains("busy") == true, topic)
            }
            for flag in ["--standard", "--compact"] {
                XCTAssertTrue(pages.pages["install"]?.contains(flag) == true, flag)
            }
            XCTAssertTrue(pages.overview.contains("install ID [--standard|--compact]"))
        }
    }

    /// **`set` is documented**, in both languages: its answer and its page.
    func testSetIsDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            XCTAssertTrue(pages.pages["set"]?.contains("\"previous\"") == true)
            XCTAssertNotNil(RemoteControlHelp.answer(["set", "--help"], pages)?["help"] as? String)
        }
    }
}

/// **What `silicontrol` can do that the Settings window does** is documented, in both languages: the switches in the
/// overview and on `set`'s page, the window's panes on `settings`' page.
final class RemoteControlSettingsHelpTests: XCTestCase {
    func testTheSwitchesAndTheSettingsAreDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            for line in ["set beta [OPTION on|off]", "set save-images [FOLDER|off]", "settings ["] {
                XCTAssertTrue(pages.overview.contains(line), line)
            }
            for option in ["more-solvers", "steps", "size-scales", "negative-lora"] {
                XCTAssertTrue(pages.pages["set"]?.contains(option) == true, option)
            }
            XCTAssertTrue(pages.pages["errors"]?.contains("remote") == true)
            XCTAssertNotNil(RemoteControlHelp.answer(["settings", "--help"], pages)?["help"] as? String)
            // The window's gestures that came to the command line last: each in the overview and on its page.
            for (topic, page, overview) in [("set", "set preview [on|off]", "set preview [on|off]"),
                                            ("rack", "rack top|dit|fold|unfold|detach|dock", "rack top|dit|fold|unfold|detach|dock"),
                                            ("remove", "remove g2", "remove g2"), ("again", "reuse <i7|g2|", "reuse <i7|g2|"),
                                            ("uninstall", "uninstall downloads|legacy [--yes]", "|downloads|legacy [--yes]")] {
                XCTAssertTrue(pages.pages[topic]?.contains(page) == true, "\(topic): \(page)")
                XCTAssertTrue(pages.overview.contains(overview), overview)
            }
            for pane in ["general", "macs", "advanced"] {
                XCTAssertTrue(pages.pages["settings"]?.contains(pane) == true, pane)
            }
        }
    }
}
