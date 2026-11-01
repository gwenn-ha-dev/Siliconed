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
            for line in ["install writer", "rewrite \"<", "describe <image|i7>", "set neural-engine on|off", "set beta [OPTION on|off]"] {
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
}
