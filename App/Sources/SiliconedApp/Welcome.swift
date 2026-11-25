// The first launch, the waiting states and the progress indicator: what the canvas shows
// before there is an image. Split from `MainView.swift`.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// **The first launch**: nothing comes with the app, so the canvas says so and offers
/// the models in front side by side (`AppState.frontFamilies`) — what each does, its license, its
/// place on disk. « Install… » opens the Models sheet on that model's license: nothing is downloaded
/// before it is accepted.
struct Welcome: View {
    let app: AppState
    /// The version chosen on each card (`Family.preselectedVariant` until the user picks another).
    @State private var choice: [Family: Variant] = [:]

    var body: some View {
        // Whole where the canvas holds it; on a small screen it scrolls rather than lose a line.
        ViewThatFits(in: .vertical) {
            offer
            ScrollView { offer.frame(maxWidth: .infinity) }.scrollBounceBehavior(.basedOnSize)
        }
    }

    private var offer: some View {
        VStack(spacing: 22) {
            VStack(spacing: 8) {
                Text("Welcome to Siliconed").font(.largeTitle.weight(.semibold))
                Text("Choose a model to begin. Its license and the space it takes are shown before anything is downloaded.")
                    .font(.callout).foregroundStyle(Palette.secondary)
                    .frame(maxWidth: 460).fixedSize(horizontal: false, vertical: true)
            }
            // Side by side when the canvas holds them, else the first two above the others.
            ViewThatFits(in: .horizontal) {
                row(AppState.frontFamilies)
                VStack(spacing: 16) {
                    row(Array(AppState.frontFamilies.prefix(2)))
                    row(Array(AppState.frontFamilies.dropFirst(2)))
                }
            }
            if let title = app.libraryJob {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(title).font(.callout).foregroundStyle(Palette.secondary)
                }
            } else {
                Button("All Models…") { app.managementOpen = true }.buttonStyle(.chartLink)
            }
        }
        .multilineTextAlignment(.center)
    }

    /// **A row of cards, all as tall as its tallest**: a card grows with its words — a long license, a version line
    /// that wraps in the user's language — instead of cutting them.
    private func row(_ families: [Family]) -> some View {
        HStack(alignment: .top, spacing: 16) { ForEach(families, id: \.self) { card($0) } }
            .fixedSize(horizontal: false, vertical: true)
    }

    private func card(_ family: Family) -> some View {
        let model = ModelCard.of(family)
        let variant = choice[family] ?? family.preselectedVariant
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Image(systemName: Self.symbol(family))
                    .font(.title2).foregroundStyle(Palette.live)
                Spacer()
                Text(verbatim: model.licenseName).font(.caption).foregroundStyle(Palette.secondary)
                    .multilineTextAlignment(.trailing).fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 6) {
                Text(verbatim: model.name).font(.title3.weight(.semibold))
                if family.isBeta { BetaBadge() }
            }
            Text(Self.blurb(family))
                .font(.callout).foregroundStyle(Palette.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            // Standard or Compact, each with its space on disk (`Family.installedSize`).
            if family.variants.count > 1 {
                VersionChoice(family: family, variant: Binding(get: { variant }, set: { choice[family] = $0 }))
                    .font(.caption)
            }
            HStack {
                if family.variants.count == 1 {
                    Label("\(Figures.disk(family.installedSize(variant))) on disk", systemImage: "internaldrive")
                        .font(.caption).foregroundStyle(Palette.secondary)
                        .help("The space it takes on disk once installed")
                }
                Spacer()
                Button("Install…") { app.offerInstall(family, variant: variant) }
                    .buttonStyle(.prominent)
                    .unavailable(app.managementRefusal)
            }
        }
        .multilineTextAlignment(.leading)
        .padding(16)
        .frame(width: 270)
        .frame(minHeight: family.variants.count > 1 ? 270 : 200, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Palette.surface))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Palette.hairline))
    }

    static func symbol(_ family: Family) -> String { family.traits.presentation.symbol }

    /// **What the model does, in one sentence** (`FamilyPresentation.blurb`). Also the tooltip of its row in the
    /// Models sheet.
    static func blurb(_ family: Family) -> String { String(localized: family.traits.presentation.blurb) }
}

/// Before the first preview: the frame of the coming image, the wheel and the current stage. No
/// animated glint: it duplicated the wheel, and redrew the view at 30 frames/s during the computation.
struct Waiting: View {
    let stage: Engine.Stage?

    var body: some View {
        ZStack {
            Palette.well
            // The stage in words only: the spinner of the render is in the progress under this frame.
            Text(stage?.label ?? String(localized: "Preparing…"))
                .font(.callout).foregroundStyle(Palette.secondary)
        }
    }
}

/// **The progress of the running render**: the stage, the step, the batch, the remaining time, one segment
/// per DiT evaluation, and "Stop".
struct ProgressIndicator: View {
    let app: AppState

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(label).font(.callout).lineLimit(1)
                    Spacer(minLength: 12)
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        // What compiles is said once, in the window's band (`CompilationMark`); here, why no time is said.
                        if app.compilingCoreAI {
                            Text("Estimate after compilation").font(.callout).foregroundStyle(Palette.secondary)
                        } else if let r = app.remaining(at: context.date) {
                            Text("\(Figures.duration(r)) left")
                                .font(.callout).monospacedDigit().foregroundStyle(Palette.secondary)
                                .contentTransition(.numericText())
                        }
                    }
                }
                if app.stage == .denoising, app.totalSteps > 0, app.totalSteps <= 60 {
                    HStack(spacing: 3) {
                        ForEach(1...app.totalSteps, id: \.self) { i in
                            Capsule()
                                .fill(i <= app.currentStep ? Palette.live : Palette.well)
                                .frame(height: 4)
                        }
                    }
                    .accessibilityHidden(true)
                    .animation(Motion.pick(.easeOut(duration: 0.3)), value: app.currentStep)
                } else {
                    ProgressView(value: app.fraction).progressViewStyle(.linear).tint(Palette.live)
                }
            }
            .frame(width: 360)
            // One element: what the render is doing, and how far it is.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(Text(verbatim: label))
            .accessibilityValue(Text(verbatim: Figures.percent(Int(app.fraction * 100))))
            Button(role: .destructive) { app.cancel() } label: {
                Image(systemName: "stop.fill")
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.borderless)
            .imageScale(.large)
            .help("Stop (⌘.)")
            .accessibilityLabel(Text("Stop"))
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .glass(RoundedRectangle(cornerRadius: 16))
    }

    private var label: String {
        var pieces: [String] = []
        if let t = app.currentJob { pieces.append(t.settings.modelName) }
        if app.batchSize > 1 {
            pieces.append(String(localized: "image \(app.batchImage + 1)/\(app.batchSize)"))
        }
        if app.stage == .denoising, app.totalSteps > 0 {
            pieces.append(String(localized: "step \(app.currentStep)/\(app.totalSteps)"))
        } else {
            pieces.append(app.stage?.label ?? String(localized: "Preparing…"))
        }
        return pieces.joined(separator: " · ")
    }
}
