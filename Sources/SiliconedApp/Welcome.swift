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
    /// The version chosen on each card (`Family.preselectedVariant` until the user picks another:
    /// Standard, Light on a Mac of 8 GB).
    @State private var choice: [Family: Variant] = [:]

    var body: some View {
        VStack(spacing: 22) {
            VStack(spacing: 8) {
                Text("Welcome to Siliconed").font(.largeTitle.weight(.semibold))
                Text("Choose a model to begin. Its license and the space it takes are shown before anything is downloaded.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: 460)
            }
            // Side by side when the canvas holds them, else two by two.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) { ForEach(AppState.frontFamilies, id: \.self) { card($0) } }
                VStack(spacing: 16) {
                    HStack(alignment: .top, spacing: 16) { ForEach(AppState.frontFamilies.prefix(2), id: \.self) { card($0) } }
                    HStack(alignment: .top, spacing: 16) { ForEach(AppState.frontFamilies.dropFirst(2), id: \.self) { card($0) } }
                }
            }
            if let title = app.forgeInProgress {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(title).font(.callout).foregroundStyle(.secondary)
                }
            } else {
                Button("All Models…") { app.managementOpen = true }.buttonStyle(.link)
            }
        }
        .multilineTextAlignment(.center)
    }

    private func card(_ family: Family) -> some View {
        let model = ModelCard.of(family)
        let variant = choice[family] ?? family.preselectedVariant()
        let size = family.installedSize(variant)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: Self.symbol(family))
                    .font(.title2).foregroundStyle(Color.accentColor)
                Spacer()
                Text(verbatim: model.licenseName).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.tail)
            }
            HStack(spacing: 6) {
                Text(verbatim: model.name).font(.title3.weight(.semibold))
                if family.isBeta { BetaBadge() }
            }
            Text(Self.blurb(family))
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            // Standard, Compact or Light, each with its space on disk (`Family.installedSize`): the rack
            // shows another figure, the memory a render needs (`ModelCard.memoryNeed`).
            if family.variants.count > 1 {
                VersionChoice(family: family, variant: Binding(get: { variant }, set: { choice[family] = $0 }))
                    .font(.caption)
            }
            HStack {
                if family.variants.count == 1 {
                    Label("\(Figures.disk(size.dit + size.encoder + size.components)) on disk", systemImage: "internaldrive")
                        .font(.caption).foregroundStyle(.secondary)
                        .help("The space it takes on disk once installed")
                }
                Spacer()
                Button("Install…") { app.offerInstall(family, variant: variant) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!app.managementPossible)
            }
        }
        .multilineTextAlignment(.leading)
        .padding(16)
        .frame(width: 270, height: family.variants.count > 1 ? 270 : 200)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Palette.surface))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Palette.hairline))
    }

    static func symbol(_ family: Family) -> String {
        switch family {
        case .qwenImage21: "wand.and.stars"
        case .anima: "paintbrush"
        case .krea2: "camera.aperture"
        case .mossImage: "text.viewfinder"
        default: "sparkles"
        }
    }

    /// **What the model does, in one sentence** — said, not sold. Also the tooltip of its row in the
    /// Models sheet.
    static func blurb(_ family: Family) -> String {
        switch family {
        case .zImage: String(localized: "Fast and photographic. The one to start with.")
        case .qwenImage21: String(localized: "Generates, and edits a photo by instruction: “change her jacket to red”.")
        // Its publisher trained it for illustration and keeps it there: no photorealism, by choice.
        case .anima: String(localized: "Illustration, painting and anime. No photorealism: its publisher's choice.")
        // The largest DiT of the four (12.8 B), and the slowest render.
        case .krea2: String(localized: "Aesthetic images, from photography to painting. The largest model, and the slowest.")
        // A 9 B language model reads the prompt; the eight steps come from the community's distillation, not the publisher's.
        case .mossImage: String(localized: "Long, detailed prompts, read by a 9 B language model. Eight steps through the community's distillation.")
        }
    }
}

/// Before the first preview: the frame of the coming image, the wheel and the current stage. No
/// animated glint: it duplicated the wheel, and redrew the view at 30 frames/s during the computation.
struct Waiting: View {
    let stage: Engine.Stage?

    var body: some View {
        ZStack {
            Palette.well
            VStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(stage?.label ?? String(localized: "Preparing…"))
                    .font(.callout).foregroundStyle(.secondary)
            }
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
                    if app.remaining(at: .now) != nil {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text("\(Figures.duration(app.remaining(at: context.date) ?? 0)) left")
                                .font(.callout).monospacedDigit().foregroundStyle(.secondary)
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
                    .animation(.easeOut(duration: 0.3), value: app.currentStep)
                } else {
                    ProgressView(value: app.fraction).progressViewStyle(.linear)
                }
            }
            .frame(width: 360)
            Button(role: .destructive) { app.cancel() } label: {
                Image(systemName: "stop.fill")
            }
            .buttonStyle(.borderless)
            .imageScale(.large)
            .help("Stop (⌘.)")
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
